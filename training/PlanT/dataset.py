import os
import sys
import copy
import glob
import logging
import json
import math
import cv2
import numpy as np
from pathlib import Path
from beartype import beartype
from einops import rearrange

import torch
from torch.utils.data import Dataset
from util.viz_tokens_bev import create_BEV


class PlanTDataset(Dataset):
    @beartype
    def __init__(self, root: str, cfg, shared_dict=None, split: str = "all") -> None:
        self.cfg = cfg
        self.cfg_train = cfg.model.training
        self.data_cache = shared_dict
        self.cnt = 0

        self.input_sequence_files = []
        self.output_sequence_files = []
        self.labels = []
        self.measurements = []

        label_raw_path_all = glob.glob(root + "/**/Routes*", recursive=True)
        label_raw_path = []

        label_raw_path = self.filter_data_by_town(label_raw_path_all, split)
            
        # logging.info(f"Found {len(label_raw_path)} Route folders")

        # add multiple datasets (different seeds while collecting data)
        if cfg.trainset_size >= 2:
            add_data_path = root[:-2] + "_2"
            label_add_path_all = glob.glob(
                add_data_path + "/**/Routes*", recursive=True
            )
            label_add_path = self.filter_data_by_town(label_add_path_all, split)
            label_raw_path += label_add_path
        if cfg.trainset_size >= 3:
            add_data_path = root[:-2] + "_3"
            label_add_path_all = glob.glob(
                add_data_path + "/**/Routes*", recursive=True
            )
            label_add_path = self.filter_data_by_town(label_add_path_all, split)
            label_raw_path += label_add_path
        if cfg.trainset_size >= 4:
            raise NotImplementedError

        logging.info(f"Found {len(label_raw_path)} Route folders containing {cfg.trainset_size} datasets.")

        for sub_route in label_raw_path:

            root_files = os.listdir(sub_route)
            routes = [
                folder
                for folder in root_files
                if not os.path.isfile(os.path.join(sub_route, folder))
            ]
            for route in routes:
                route_dir = Path(f"{sub_route}/{route}")
                num_seq = len(os.listdir(route_dir / "boxes"))

                # ignore the first 5 and last two frames
                for seq in range(
                    5,
                    num_seq - self.cfg_train.pred_len - self.cfg_train.seq_len - 2,
                ):
                    # load input seq and pred seq jointly
                    label = []
                    measurement = []
                    for idx in range(
                        self.cfg_train.seq_len + self.cfg_train.pred_len
                    ):
                        labels_file = route_dir / "boxes" / f"{seq + idx:04d}.json"
                        measurements_file = (
                            route_dir / "measurements" / f"{seq + idx:04d}.json"
                        )
                        label.append(labels_file)
                        measurement.append(measurements_file)

                    self.labels.append(label)
                    self.measurements.append(measurement)

        # There is a complex "memory leak"/performance issue when using Python objects like lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.labels       = np.array(self.labels      ).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)
        print(f"Loading {len(self.labels)} samples from {len(root)} folders")


    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.measurements)


    def __getitem__(self, index):
        """Returns the item at index idx."""

        labels = self.labels[index]
        measurements = self.measurements[index]

        sample = {
            "input": [],
            "output": [],
            "brake": [],
            "waypoints": [],
            "target_point": [],
            "light": [],
        }
        if not self.data_cache is None and labels[0] in self.data_cache:
            sample = self.data_cache[labels[0]]
        else:
            loaded_labels = []
            loaded_measurements = []

            for i in range(self.cfg_train.seq_len + self.cfg_train.pred_len):
                measurements_i = json.load(open(measurements[i]))
                labels_i = json.load(open(labels[i]))

                loaded_labels.append(labels_i)
                loaded_measurements.append(measurements_i)

            # ego car is always the first one in label file
            waypoints = get_waypoints(loaded_measurements[self.cfg_train.seq_len - 1 :])
            waypoints = transform_waypoints(waypoints)

            # save waypoints in meters
            filtered_waypoints = []
            for id in ["1"]:
                waypoint = []
                for matrix, _ in waypoints[id][1:]:
                    waypoint.append(matrix[:2, 3])
                filtered_waypoints.append(waypoint)
            waypoints = np.array(filtered_waypoints)

            ego_waypoint = waypoints[-1]

            sample["waypoints"] = ego_waypoint
      
            local_command_point = np.array(loaded_measurements[self.cfg_train.seq_len - 1]["target_point"])
            sample["target_point"] = tuple(local_command_point)
            sample["light"] = loaded_measurements[self.cfg_train.seq_len - 1][
                "light_hazard"
            ]

            if self.cfg.model.pre_training.pretraining == "forecast":
                offset = (
                    self.cfg.model.pre_training.future_timestep
                )  # target is next timestep
            elif self.cfg.model.pre_training.pretraining == "none":
                offset = 0
            else:
                print(
                    f"ERROR: pretraining {self.cfg.model.pre_training.pretraining} is not supported"
                )
                sys.exit()

            for sample_key, file in zip(
                ["input", "output"],
                [
                    (
                        loaded_measurements[self.cfg_train.seq_len - 1],
                        loaded_labels[self.cfg_train.seq_len - 1],
                    ),
                    (
                        loaded_measurements[self.cfg_train.seq_len - 1 + offset],
                        loaded_labels[self.cfg_train.seq_len - 1 + offset],
                    ),
                ],
            ):

                measurements_data = file[0]
                ego_matrix = np.array(measurements_data["ego_matrix"])
                ego_yaw = measurements_data['theta']
                if sample_key == "input":
                    ego_matrix_input = ego_matrix
                    ego_yaw_input = ego_yaw
                    
                labels_data_all = file[1]

                if self.cfg_train.input_ego:
                    labels_data = file[1]
                else:
                    labels_data = file[1][1:]  # remove ego car

                # for future timesteps transform position to ego frame of first timestep
                pos = []
                yaw = []
                for labels_data_i in labels_data:
                    p = np.array(copy.deepcopy(labels_data_i["position"])) - np.array(labels_data_all[0]["position"])
                    p = np.append(p,[1])
                    p[1] = -p[1]
                    p_global = ego_matrix @ p
                    p_t2 = np.linalg.inv(ego_matrix_input) @ p_global
                    p_t2[1] = -p_t2[1]
                    pos.append(p_t2[:2])
                    yaw.append(labels_data_i["yaw"]+ego_yaw-ego_yaw_input)
                
                data_car = [
                    [
                        1.0,  # type indicator for cars
                        float(pos[j][0]),
                        float(pos[j][1]),
                        float(yaw[j] * 180 / 3.14159265359),  # in degrees
                        float(x["speed"] * 3.6),  # in km/h
                        float(x["extent"][2]),
                        float(x["extent"][1]),
                        float(x["id"]),
                    ]
                    for j, x in enumerate(labels_data)
                    if x["class"] == "Car" and ((self.cfg_train.remove_back and float(pos[j][0]) >= 0) or not self.cfg_train.remove_back)
                ]

                if sample_key == "output":
                    # discretize box
                    if self.cfg.model.pre_training.quantize:
                        if len(data_car) > 0:
                            data_car = self.quantize_box(data_car)

                    if self.cfg.model.pre_training.pretraining == "forecast":
                        # we can only use vehicles where we have the corresponding object in the input timestep
                        # if we don't have the object in the input timestep, we remove the vehicle
                        # if we don't have the object in the output timestep we add a dummy vehicle, that is not considered for the loss
                        data_car_by_id = {}
                        i = 0
                        for ii, x in enumerate(labels_data):
                            if x["class"] == "Car" and ((self.cfg_train.remove_back and float(pos[ii][0]) >= 0) or not self.cfg_train.remove_back):
                                data_car_by_id[x["id"]] = data_car[i]
                                i += 1
                        data_car_matched = []
                        for i, x in enumerate(data_car_input):
                            input_id = x[7]
                            if input_id in data_car_by_id:
                                data_car_matched.append(data_car_by_id[input_id])
                            else:
                                # append dummy data
                                dummy_data = x
                                dummy_data[0] = 10.0  # type indicator for dummy
                                data_car_matched.append(dummy_data)

                        data_car = data_car_matched
                        assert len(data_car) == len(data_car_input)

                else:
                    data_car_input = data_car

                # remove id from data_car
                data_car = [x[:-1] for x in data_car]

                # if we use the far_node as target waypoint we need the route as input
                data_route = [
                    [
                        2.0,  # type indicator for route
                        float(x["position"][0]) - float(labels_data_all[0]["position"][0]),
                        float(x["position"][1]) - float(labels_data_all[0]["position"][1]),
                        float(x["yaw"] * 180 / 3.14159265359),  # in degrees
                        float(x["id"]),
                        float(x["extent"][2]),
                        float(x["extent"][1]),
                    ]
                    for j, x in enumerate(labels_data)
                    if x["class"] == "Route"
                    and float(x["id"]) < self.cfg_train.max_NextRouteBBs
                ]
                
                # we split route segment slonger than 10m into multiple segments
                # improves generalization
                data_route_split = []
                for route in data_route:
                    if route[6] > 10:
                        routes = split_large_BB(
                            route, len(data_route_split)
                        )
                        data_route_split.extend(routes)
                    else:
                        data_route_split.append(route)
                data_route = data_route_split[: self.cfg_train.max_NextRouteBBs]

                if sample_key == "output":
                    data_route = data_route[: len(data_route_input)]
                    if len(data_route) < len(data_route_input):
                        diff = len(data_route_input) - len(data_route)
                        data_route.extend([data_route[-1]] * diff)
                else:
                    data_route_input = data_route


                if self.cfg.model.training.remove_velocity == 'input':
                    if sample_key == 'input':
                        for i in range(len(data_car)):
                            data_car[i][4] = 0.
                elif self.cfg.model.training.remove_velocity == 'None':
                    pass
                else:
                    raise NotImplementedError
                
                if self.cfg.model.training.route_only_wp == True:
                    if sample_key == 'input':
                        for i in range(len(data_route)):
                            data_route[i][3] = 0.
                            data_route[i][-2] = 0.
                            data_route[i][-1] = 0.
                elif self.cfg.model.training.route_only_wp == False:
                    pass
                else:
                    raise NotImplementedError


                assert len(data_route) == len(
                    data_route_input
                ), "Route and route input not the same length"

                assert (
                    len(data_route) <= self.cfg_train.max_NextRouteBBs
                ), "Too many routes"

                if len(data_route) == 0:
                    # quit programm
                    print("ERROR: no route found")
                    logging.error("No route found in file: {}".format(file))
                    sys.exit()

                sample[sample_key] = data_car + data_route
                
                # for debugging, need to also uncomment saving inside of the function
                if self.cfg.visualize:
                    create_BEV(data_car, data_route, True, False, inp=sample_key, cnt=self.cnt, visualize=True)

            if not self.data_cache is None:
                self.data_cache[labels[0]] = sample

        assert len(sample["input"]) == len(
            sample["output"]
        ), "Input and output have different length"

        self.cnt+=1
        return sample

    def quantize_box(self, boxes):
        boxes = np.array(boxes)

        # range of xy is [-30, 30]
        # range of yaw is [-360, 0]
        # range of speed is [0, 60]
        # range of extent is [0, 30]

        # quantize xy
        boxes[:, 1] = (boxes[:, 1] + 30) / 60
        boxes[:, 2] = (boxes[:, 2] + 30) / 60

        # quantize yaw
        boxes[:, 3] = (boxes[:, 3] + 360) / 360

        # quantize speed
        boxes[:, 4] = boxes[:, 4] / 60

        # quantize extent
        boxes[:, 5] = boxes[:, 5] / 30
        boxes[:, 6] = boxes[:, 6] / 30

        boxes[:, 1:] = np.clip(boxes[:, 1:], 0, 1)

        size_pos = pow(2, self.cfg.model.pre_training.precision_pos)
        size_speed = pow(2, self.cfg.model.pre_training.precision_speed)
        size_angle = pow(2, self.cfg.model.pre_training.precision_angle)

        boxes[:, [1, 2, 5, 6]] = (boxes[:, [1, 2, 5, 6]] * (size_pos - 1)).round()
        boxes[:, 3] = (boxes[:, 3] * (size_angle - 1)).round()
        boxes[:, 4] = (boxes[:, 4] * (size_speed - 1)).round()

        return boxes.astype(np.int32).tolist()


    def filter_data_by_town(self, label_raw_path_all, split):
        # in case we want to train without T2 and T5
        label_raw_path = []
        if split == "train":
            for path in label_raw_path_all:
                if "Town02" in path or "Town05" in path:
                    continue
                label_raw_path.append(path)
        elif split == "val":
            for path in label_raw_path_all:
                if "Town02" in path or "Town05" in path:
                    label_raw_path.append(path)
        elif split == "all":
            label_raw_path = label_raw_path_all
            
        return label_raw_path

def split_large_BB(route, start_id):
    x = route[1]
    y = route[2]
    angle = route[3] - 90
    extent_x = route[5] / 2
    extent_y = route[6] / 2

    x1 = x - extent_y * math.sin(math.radians(angle))
    y1 = y - extent_y * math.cos(math.radians(angle))

    x0 = x + extent_y * math.sin(math.radians(angle))
    y0 = y + extent_y * math.cos(math.radians(angle))

    number_of_points = (
        math.ceil(extent_y * 2 / 10) - 1
    )  # 5 is the minimum distance between two points, we want to have math.ceil(extent_y / 5) and that minus 1 points
    xs = np.linspace(
        x0, x1, number_of_points + 2
    )  # +2 because we want to have the first and last point
    ys = np.linspace(y0, y1, number_of_points + 2)

    splitted_routes = []
    for i in range(len(xs) - 1):
        route_new = route.copy()
        route_new[1] = (xs[i] + xs[i + 1]) / 2
        route_new[2] = (ys[i] + ys[i + 1]) / 2
        route_new[4] = float(start_id + i)
        route_new[5] = extent_x * 2
        route_new[6] = route[6] / (
            number_of_points + 1
        )
        splitted_routes.append(route_new)

    return splitted_routes


def get_waypoints(measurements):
    assert len(measurements) == 5
    num = 5
    waypoints = {"1": []}

    for i in range(0, num):
        waypoints["1"].append([measurements[i]["ego_matrix"], True])

    Identity = list(list(row) for row in np.eye(4))
    # padding here
    for k in waypoints.keys():
        while len(waypoints[k]) < num:
            waypoints[k].append([Identity, False])
    return waypoints


def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""

    # TODO should transform to virtual lidar coordicate?
    T = get_vehicle_to_virtual_lidar_transform()

    for k in waypoints.keys():
        vehicle_matrix = np.array(waypoints[k][0][0])
        vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
        for i in range(1, len(waypoints[k])):
            matrix = np.array(waypoints[k][i][0])
            waypoints[k][i][0] = T @ vehicle_matrix_inv @ matrix

    return waypoints


def get_virtual_lidar_to_vehicle_transform():
    # This is a fake lidar coordinate
    T = np.eye(4)
    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T


def get_vehicle_to_virtual_lidar_transform():
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())


def generate_batch(data_batch):
    input_batch, output_batch = [], []
    for element_id, sample in enumerate(data_batch):
        input_item = torch.tensor(sample["input"], dtype=torch.float32)
        output_item = torch.tensor(sample["output"])

        input_indices = torch.tensor([element_id] * len(input_item)).unsqueeze(1)
        output_indices = torch.tensor([element_id] * len(output_item)).unsqueeze(1)

        input_batch.append(torch.cat([input_indices, input_item], dim=1))
        output_batch.append(torch.cat([output_indices, output_item], dim=1))

    waypoints_batch = torch.tensor([sample["waypoints"] for sample in data_batch])
    tp_batch = torch.tensor(
        [sample["target_point"] for sample in data_batch], dtype=torch.float32
    )
    light_batch = rearrange(
        torch.tensor([sample["light"] for sample in data_batch]), "b -> b 1"
    )

    return input_batch, output_batch, waypoints_batch, tp_batch, light_batch
