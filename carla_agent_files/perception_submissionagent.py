import os
import math
import json
import time
import itertools
from copy import deepcopy
from PIL import Image, ImageDraw, ImageOps
from collections import deque
from munkres import Munkres
from pathlib import Path
from rdp import rdp

import carla
import cv2
import numpy as np
import torch

from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString

from leaderboard.autoagents import autonomous_agent
from training.Perception.model import LidarCenterNet
from training.Perception.data import lidar_to_histogram_features, draw_target_point
from training.Perception.config import GlobalConfig
from carla_agent_files.agent_utils import transfuser_utils
from scipy.interpolate import UnivariateSpline
from carla_agent_files.agent_utils.coordinate_utils import normalize_angle


DATA_SAVE_PATH = os.environ.get('DATA_SAVE_PATH')

if not DATA_SAVE_PATH:
    DATA_SAVE_PATH = None
else:
    Path(DATA_SAVE_PATH).mkdir(parents=True, exist_ok=True)

class PerceptionAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        self.config_path = path_to_conf_file
        self.step = -1
        self.lidar_freq = 1.0/ 10.0 #In seconds
        self.simulator_time_step = (1.0/20.0)
        self.max_num_bb_forecast = 4 # Number of consecutive bb detection needed for a forecast
        self.min_num_bb_forecast = 4 # Minimum number of consecutive bb detection needed for a forecast
        self.bb_buffer = deque(maxlen=self.max_num_bb_forecast)
        for i in range(self.max_num_bb_forecast - self.min_num_bb_forecast):
            self.bb_buffer.append([]) #Fill in empty bounding boxes for the optional timesteps
        self.initialized = False
        
        self.control = carla.VehicleControl()
        self.control.steer = 0.0
        self.control.throttle = 0.0
        self.control.brake = 1.0

        args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
        self.args = json.load(args_file)
        args_file.close()
        self._vehicle = None
        self.ego_pos_buffer = deque(maxlen=10)
        self.label_raw_tf_new_previous = None
        self.cfg = None
        
        # self.cfg.route_buffer = True
        # self.cfg.route_num_wp_preds = 10
        
                
        # route buffer
        self.route_buffer = []

        # setting machine to avoid loading files
        self.config = GlobalConfig(setting='eval')
        # Overwrite all properties that were set in the save config (args.txt) -> training paramters.
        self.config.__dict__.update(self.args)

        image_architecture = self.args['image_architecture']
        lidar_architecture = self.args['lidar_architecture']
        use_velocity = bool(self.args['use_velocity'])

        self.backbone = self.args['backbone']  # Options 'geometric_fusion', 'transFuser', 'late_fusion', 'latentTF'
        # self.backbone = 'transFuser'  # Options 'geometric_fusion', 'transFuser', 'late_fusion', 'latentTF'

        self.lidar_pos = self.config.lidar_pos  # x, y, z coordinates of the LiDAR position.
        # NMS only for ensambles -> not implemented yet
        self.iou_treshold_nms = self.config.iou_treshold_nms # Iou threshold used for Non Maximum suppression on the Bounding Box predictions.

        # Load model files
        self.nets = []
        self.model_count = 0 # Counts how many models are in our ensemble
        for file in os.listdir(path_to_conf_file):
            if file.endswith(".pth"):
                self.model_count += 1
                print(os.path.join(path_to_conf_file, file))
                net = LidarCenterNet(self.config, 'cuda', self.backbone, image_architecture, lidar_architecture, use_velocity)
                if(self.config.sync_batch_norm == True):
                    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net) # Model was trained with Sync. Batch Norm. Need to convert it otherwise parameters will load incorrectly.
                state_dict = torch.load(os.path.join(path_to_conf_file, file), map_location='cuda:0')
                state_dict = {k[7:]: v for k, v in state_dict.items()} # Removes the .module coming from the Distributed Training. Remove this if you want to evaluate a model trained without DDP.
                net.load_state_dict(state_dict, strict=False)
                net.cuda()
                net.eval()
                self.nets.append(net)

        self.aug_degrees = [0] # Test time data augmentation. Unused we only augment by 0 degree.
        

    @torch.inference_mode() # Faster version of torch_no_grad
    def run_step(self, input_data, tick_data, state_log, label_raw, gt_traffic_light_hazard=False, route_gt_map=None):
        self.ego_extent = label_raw[0]['extent']
        
        self.state_log = state_log
        self.route_gt_map = route_gt_map
        self.step += 1

        self.update_bb_buffer(self.bb_buffer)
        if self.cfg.route_buffer:
            self.ego_pos_buffer.append(carla.Location(x=tick_data['gps'][0],y=tick_data['gps'][1]))
        
        if (self.step+1)%2 == 1 and self.step > 0: # Only update the gps buffer every other step.
            return self.label_raw_tf, self.pred_traffic_light

        # prepare image input
        image = self.prepare_image(tick_data)

        num_points = None
        # prepare LiDAR input
        if (self.config.use_point_pillars == True):
            lidar_cloud = deepcopy(input_data['lidar'][1])
            lidar_cloud[:, 1] *= -1  # invert
            lidar_bev = [torch.tensor(lidar_cloud).to('cuda', dtype=torch.float32)]
            num_points = [torch.tensor(len(lidar_cloud)).to('cuda', dtype=torch.int32)]
        else:
            lidar_bev = self.prepare_lidar(tick_data)

        # prepare goal location input
        # target_point_image, target_point = self.prepare_goal_location(tick_data)
        target_point_image = None
        target_point = tick_data['target_point']

        # prepare velocity input
        gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32) # used by controller
        velocity = gt_velocity.reshape(1, 1) # used by transfuser

        pred_route_wp, rotated_bb, bboxes, pred_route, pred_traffic_light = self.nets[0].forward_ego(image, lidar_bev, target_point, target_point_image, velocity,
                                                num_points=num_points, save_path='img/',
                                                debug=self.cfg.viz,
                                                bb_confidence_threshold=0.1,
                                                tl_threshold=0.5)
        pred_route_wp = pred_route_wp[:, :self.cfg.route_num_wp_preds, ...]
        if route_gt_map is not None:
            pred_route_wp = torch.tensor(route_gt_map).unsqueeze(0)-torch.tensor([1.3,0])
        
        bounding_boxes = []
        bounding_boxes.append(rotated_bb)

        self.bs_vehicle_coordinate_system = self.non_maximum_suppression(bounding_boxes, self.iou_treshold_nms)
        self.bb_buffer.append(self.bs_vehicle_coordinate_system)
        
        self.instances = self.match_bb(self.bb_buffer) # Associate bounding boxes to instances
        self.list_of_unique_instances = [l[0] for l in self.instances]
        speed = self.get_speed()
        
        
        self.label_raw_tf = self.tf_bb_to_carla_bb(self.bs_vehicle_coordinate_system, bboxes, speed, pred_route_wp, pred_route)
        self.label_raw_tf = [label_raw[0]] + self.label_raw_tf
        
        # viz_trigger = gt_traffic_light_hazard or (gt_traffic_light_hazard != pred_traffic_light)
        viz_trigger = ((self.step+1) % 20 == 0 and self.cfg.viz)
        
        if self.cfg.route_buffer and route_gt_map is None:
            self.update_route_buffer(self.label_raw_tf, target_point)
            label_raw_tf_new = self.get_interpolated_route(self.label_raw_tf, self.route_buffer, self._vehicle)
        else:
            label_raw_tf_new = self.label_raw_tf
            
        if viz_trigger and self.step > 2:
            create_BEV(self.state_log, label_raw, self.label_raw_tf, gt_traffic_light_hazard, pred_traffic_light, image, lidar_bev, target_point, self.route_buffer, self._vehicle, self.cfg.route_buffer, route_gt_map)

        self.label_raw_tf = label_raw_tf_new
        self.pred_traffic_light = pred_traffic_light
        return self.label_raw_tf, self.pred_traffic_light
    
    def update_route_buffer(self, label_raw, target_point):
        
        #Current position of the car
        ego_x = self.state_log[-1][0]
        ego_y = self.state_log[-1][1]
        ego_theta = self.state_log[-1][2]
        
        rotation_matrix = np.array([[np.cos(ego_theta), -np.sin(ego_theta), 0],
                                    [np.sin(ego_theta), np.cos(ego_theta), 0],
                                    [0, 0, 1]])
        
        ego_pos_global = np.array([ego_x, ego_y, 0])
        
        # get global coords of route:
        if len(self.ego_pos_buffer) >= 10 and self.ego_pos_buffer[-1].distance(self.ego_pos_buffer[0]) < 0.2:
            pass
        else:
            x_target = target_point[0][0] # front positive
            y_target = target_point[0][1] # right poisitive
            for object in label_raw:
                if object['class'] == 'Route':
                    # ego_vehicle_matrix = np.array(self._vehicle.get_transform().get_matrix())
                    pos = [carla.Location(x=object['position'][0], y=object['position'][1])]
                    pos = np.array([pos[0].x, pos[0].y, pos[0].z])
                    pos[1] = -pos[1]
                    # pos [0] front positive, pos[1] right positive
                    
                    # we only use prediction roughly till the first target point, since this is more accurate
                    # if target is in front (center) and route is in frotn of target, do not include
                    if abs(y_target) < self.cfg.cut_off_threshold_side and pos[0] - x_target > 1: 
                        continue
                    # if target is in right and route is in right of target, do not include
                    if y_target > 1 and pos[1] - y_target > 1: 
                        continue
                    # if target is in left and route is in left of target, do not include
                    if y_target < -1 and pos[1] - y_target < -1: 
                        continue
                    
                    rot2 = rotation_matrix#[:3, :3]
                    relative_pos2 = rot2 @ pos #[:2]
                    route_pos_global = relative_pos2 + ego_pos_global

                    # rot = ego_vehicle_matrix[:3, :3]
                    # relative_pos = rot @ pos
                    # route_pos_global = relative_pos + ego_vehicle_matrix[:3, 3]
                    self.route_buffer.append([route_pos_global, object['id']])
    
    
    def get_interpolated_route(self, label_raw_tf, route_buffer, _vehicle):
        
        label_raw_tf_new = [object for object in label_raw_tf if object['class'] == 'Car']
        
        x_all = []
        y_all = []
        # ego_extent = label_raw_tf[0]['extent']
        # ego_extent = self._vehicle.bounding_box.extent
        
        ego_x = self.state_log[-1][0]
        ego_y = self.state_log[-1][1]
        ego_theta = self.state_log[-1][2]
        rotation_matrix = np.array([[np.cos(ego_theta), -np.sin(ego_theta), 0],
                                    [np.sin(ego_theta), np.cos(ego_theta), 0],
                                    [0, 0, 1]])
        
        ego_pos_global = np.array([ego_x, ego_y, 0])
        

        for i, route in enumerate(route_buffer):
            # ego_vehicle_matrix = np.array(_vehicle.get_transform().get_matrix())
            global_route = route[0] #np.array([route[0].x, route[0].y, route[0].z])
            # local_route_pos = global_route - ego_vehicle_matrix[:3, 3]
            local_route_pos = global_route - ego_pos_global
            rot = rotation_matrix[:3, :3]
            relative_route = rot.T @ local_route_pos
            x = relative_route[0]
            y = -relative_route[1]
            
            # cut off route if it is too far away or behind the ego vehicle
            if relative_route[0] > -5.0 and np.sqrt(relative_route[1]**2+relative_route[0]**2) < 35:
                x_all.append(x)
                y_all.append(y)

            # remove points which we already passed
            if relative_route[0] < -10.0:
                self.route_buffer.pop(i)

        x_all, y_all = zip(*sorted(zip(x_all, y_all)))
        
        x_all = np.array(x_all)
        y_all = np.array(y_all)

        s = x_all.shape[0]*100
        if x_all.shape[0] <= 3:
            k = x_all.shape[0]-1
        else:
            k=3
        spline = UnivariateSpline(x_all, y_all, s=s, k=k)
        max_x = x_all.max()
        if max_x < 7.5 and self.cfg.min_max_interpolation_distance:
            max_x = 7.5
        x_spline = np.arange(self.cfg.min_interpolation_distance, max_x, 0.5)
        y_spline = spline(x_spline)
        
        shortened_route = rdp(np.array(list(zip(x_spline, y_spline))), epsilon=0.5)
        
        vectors = shortened_route[1:] - shortened_route[:-1]
        midpoints = shortened_route[:-1] + vectors/2.
        try:
            norms = np.linalg.norm(vectors, axis=1)
        except:
            return self.label_raw_tf_new_previous

        angles = np.arctan2(vectors[:,1], vectors[:,0])

        for i, midpoint in enumerate(midpoints):
            # find distance to center of waypoint
            relative_pos = np.array([midpoint[0], midpoint[1], 0.0])
            transform = carla.Transform(carla.Location(x=relative_pos[0], y=relative_pos[1], z=relative_pos[2]), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
            distance = np.linalg.norm(relative_pos)
            
            # find distance to beginning of bounding box
            st_relative_pos = np.array([shortened_route[i][0], shortened_route[i][1], 0.0])
            st_distance = np.linalg.norm(st_relative_pos)

            # only store route boxes that are near the ego vehicle
            self.max_route_distance = 30.0
            
            if i > 0 and st_distance > self.max_route_distance:
                continue

            length_bounding_box = carla.Vector3D(norms[i]/2., self.ego_extent[2]/2, self.ego_extent[0]/2)
            bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
            bounding_box.rotation = carla.Rotation(pitch = 0.0,
                                                yaw   = angles[i] * 180 / np.pi,
                                                roll  = 0.0)

            route_extent = bounding_box.extent
            dx = np.array([route_extent.x, route_extent.y, route_extent.z]) * 2.
            relative_yaw = -angles[i]# - ego_yaw

            result = {
                "class": "Route",
                "extent": [dx[2], dx[0], dx[1]],
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "centre_distance": distance,
                "starting_distance": st_distance,
                "id": i,
            }
            label_raw_tf_new.append(result)
            
        if i == 0:
            label_raw_tf_new.append(result)
        self.label_raw_tf_new_previous = label_raw_tf_new
            
        return label_raw_tf_new
        
    
    def tf_bb_to_carla_bb(self, bb, bboxes, speed, pred_route_wp, pred_route):
        # transform bounding box to carla bounding box
        
        if speed == False:
            speed = [0.0]*len(bb)
        else:
            speed = speed[::-1]
        
        label_raw_tf = []
        speed_iter = 0
        for ix, box in enumerate(bb):
            if ix not in self.list_of_unique_instances:
                continue
            label_raw_tf.append({})
            label_raw_tf[-1]['class'] = 'Car'
            label_raw_tf[-1]['extent']  = [2, bboxes[ix][3].item()/8, bboxes[ix][2].item()/8]
            label_raw_tf[-1]['position'] =  [box[4][0]-1.3, -box[4][1], 0] # vehicles are predicted in vehicle coordinate system but we need it in lidar coordinate system
            label_raw_tf[-1]['yaw'] = bboxes[ix][4].item()
            label_raw_tf[-1]['speed'] = speed[speed_iter]
            label_raw_tf[-1]['id'] = ix
            speed_iter += 1
        
        if not self.cfg.route_buffer or self.route_gt_map is not None:
            # ego_extent = self._vehicle.bounding_box.extent
            # ego_extent = label_raw_tf[0]['extent']
            if self.route_gt_map is None:
                shortened_route = rdp(np.array(pred_route_wp.squeeze().cpu()), epsilon=0.5)
            else:
                shortened_route = rdp(np.array(pred_route_wp).squeeze(), epsilon=0.5)
            # convert points to vectors
            vectors = shortened_route[1:] - shortened_route[:-1]
            midpoints = shortened_route[:-1] + vectors/2.
            norms = np.linalg.norm(vectors, axis=1)
            angles = np.arctan2(vectors[:,1], vectors[:,0])
            ego_theta = self.state_log[-1][2]
            
            for i, midpoint in enumerate(midpoints):
                # find distance to center of waypoint
                relative_pos = np.array([midpoint[0], midpoint[1], 0.0])
                transform = carla.Transform(carla.Location(x=relative_pos[0], y=relative_pos[1], z=relative_pos[2]), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
                distance = np.linalg.norm(relative_pos)
                
                # find distance to beginning of bounding box
                st_relative_pos = np.array([shortened_route[i][0], shortened_route[i][1], 0.0])
                st_distance = np.linalg.norm(st_relative_pos)


                # only store route boxes that are near the ego vehicle
                self.max_route_distance = 30.0
                if i > 0 and st_distance > self.max_route_distance:
                    continue

                length_bounding_box = carla.Vector3D(norms[i]/2., self.ego_extent[2]/2, self.ego_extent[0]/2)
                bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
                bounding_box.rotation = carla.Rotation(pitch = 0.0,
                                                    yaw   = angles[i] * 180 / np.pi,
                                                    roll  = 0.0)

                route_extent = bounding_box.extent
                dx = np.array([route_extent.x, route_extent.y, route_extent.z]) * 2.
                relative_yaw = -angles[i] #- ego_theta
                
                label_raw_tf.append({})
                label_raw_tf[-1]['class'] = 'Route'
                label_raw_tf[-1]['extent']  = [dx[2], dx[0], dx[1]]
                label_raw_tf[-1]['position'] =  [relative_pos[0], relative_pos[1], relative_pos[2]]
                label_raw_tf[-1]['yaw'] = relative_yaw 
                # label_raw_tf[-1]['yaw'] = np.arctan2(box[1].item(), box[0].item()-1.3) #TODO
                label_raw_tf[-1]['id'] = i
                
        else:
            dx = np.array([2.1283, 1.5107, 1.0])
            relative_yaw = np.arctan2(0.5,0.5) # placeholder yaw, we overwrite this later anyway

            # Add route
            for ix, box in enumerate(pred_route_wp[0]):
                if self.cfg.route_buffer:
                    relative_pos = np.array([box[0].item(), -box[1].item(), 0])
                # route_prop = pred_route[0][ix]
                label_raw_tf.append({})
                label_raw_tf[-1]['class'] = 'Route'
                label_raw_tf[-1]['extent']  = [dx[2], dx[0], dx[1]]
                label_raw_tf[-1]['position'] =  [relative_pos[0], relative_pos[1], relative_pos[2]]
                label_raw_tf[-1]['yaw'] = relative_yaw # in radians
                label_raw_tf[-1]['id'] = ix
            
        return label_raw_tf

    def non_maximum_suppression(self, bounding_boxes, iou_treshhold):
        filtered_boxes = []
        bounding_boxes = np.array(list(itertools.chain.from_iterable(bounding_boxes)), dtype=np.object)

        if(bounding_boxes.size == 0): #If no bounding boxes are detected can't do NMS
            return filtered_boxes


        confidences_indices = np.argsort(bounding_boxes[:, 2])
        while (len(confidences_indices) > 0):
            idx = confidences_indices[-1]
            current_bb = bounding_boxes[idx, 0]
            filtered_boxes.append(current_bb)
            confidences_indices = confidences_indices[:-1] #Remove last element from the list

            if(len(confidences_indices) == 0):
                break

            for idx2 in deepcopy(confidences_indices):
                if(self.iou_bbs(current_bb, bounding_boxes[idx2, 0]) > iou_treshhold): # Remove BB from list
                    confidences_indices = confidences_indices[confidences_indices != idx2]

        return filtered_boxes
    
    def match_bb(self, buffer_bb):
        instances = []
        # We only start after we have 4 time steps.
        if(len(buffer_bb)  < self.max_num_bb_forecast): 
            return instances

        all_indices = []
        for i in range(len(buffer_bb) - 1):
            if (len(buffer_bb[i]) == 0 or len(buffer_bb[i+1]) == 0):
                # Timestep without bounding boxes so there is no match
                all_indices.append([])
                continue

            matrix_size = max(len(buffer_bb[i]), len(buffer_bb[i+1]))

            # Initialize with a large value so that bb that don't exist get matched last.
            ious = np.ones((matrix_size, matrix_size)) * 10.0 
            for j in range(len(buffer_bb[i])):
                for k in range(len(buffer_bb[i+1])):
                    # Invert IOU here to convert value to costs
                    ious[j, k] = 1.0 - self.iou_bbs(buffer_bb[i][j], buffer_bb[i+1][k]) 

            m = Munkres()
            indexes = m.compute(ious)
            all_indices.append(indexes)

        inv_instances = []
        # Create instance for every newest bb.
        for i in range(len(buffer_bb[-1])-1, -1, -1): 
            instance = [i]
            write = True
            continue_adding_bbs = True
            last_timestep_index = i
            # Loops over available timesteps starting with the latest
            for j in range(len(buffer_bb)-1,0,-1): 
                if(continue_adding_bbs == False):
                    break

                #There was a timestep with no matches / no bbs.
                if(len(all_indices[j-1]) == 0):
                    # If we have enough bb write the instance, else delete it.
                    if (len(instance) < self.min_num_bb_forecast):
                        write = False
                    break
                # Loops over pairs for each timestep
                for k in range(len(all_indices[j-1])): 
                    # Find the match for the current bb
                    if(all_indices[j-1][k][1] == last_timestep_index): 
                        # Check if the matched bb actually exists
                        if(all_indices[j-1][k][0] >= len(buffer_bb[j-1])): 
                            # This instance has a timestep without a bb
                            if(len(instance) >= self.min_num_bb_forecast): 
                                # Stop instance here and write it
                                continue_adding_bbs = False
                                break
                            else:
                                # There are less total bb than needed. Delete instance!
                                write = False 
                        else:
                            instance.append(all_indices[j-1][k][0])
                            last_timestep_index = all_indices[j-1][k][0]
                            break

            if(write==True):
                inv_instances.append(instance)
        return inv_instances
    
    def get_speed(self):
        # We only start after we have 4 time steps.
        if (len(self.bb_buffer) < self.max_num_bb_forecast):  
            return False
        
        speed = []

        self.instance_future_bb = []
        for i in range(len(self.instances)):
            bb_array = self.get_bb_of_instance(i)

            # 0 index is the oldest timestep
            # Ids are from old -> new
            box_m1 = bb_array[-1] # Most recent bounding box
            box_m2 = bb_array[-2]

            center_top = box_m2[4] * np.array([np.cos(box_m2[2]), np.sin(box_m2[2])]) + box_m2[0:2]
            point = self.getProjectedPointOnLine(box_m2[0:2], center_top, box_m1[0:2])[0:2]
            distance_vector_m2 = point - box_m2[0:2]
            # Our predictions happen at 100ms intervals. So we need to multiply by 10 to get m/s scale.
            velocity_m2 = np.linalg.norm(distance_vector_m2) / self.lidar_freq
            if velocity_m2 < 0.1: velocity_m2 = 0.0
            speed.append(velocity_m2)
            
        return speed
    
    # Projects the new center onto the line that the vehicle would have when driving straight.
    # Used to remove noisy sidewards movements
    def getProjectedPointOnLine(self, center, top, new_center):
        point = Point(new_center[0], new_center[1])
        line = LineString([(center[0], center[1]), (top[0], top[1])])

        x = np.array(point.coords[0])

        u = np.array(line.coords[0])
        v = np.array(line.coords[len(line.coords) - 1])

        n = v - u
        norm = np.linalg.norm(n, 2)
        if(norm < 0.000001): #Avoid division by 0
            return new_center

        n /= norm

        P = u + n * np.dot(x - u, n)
        P = np.array([P[0], P[1], 1.0])
        return P
    
    def get_bb_of_instance(self, instance_id):
        '''
        Args:
            instance_id: The instance if of the bounding box in the self.instances array
        Returns:
            List of bounding boxes belonging to that instance. The first item is the oldest bb, the last one is the most recent one.
            An instance can have a varying number of past bounding boxes, so accessing the array from back to front is advised.
            Format of BB: [x,y, orientation, speed, extent_x, extent_y]
        '''
        if (len(self.bb_buffer) < self.max_num_bb_forecast):  # We only start after we have 4 time steps.
            return []
        instance_bbs = []
        for j in range(self.max_num_bb_forecast):      # From oldest to newest BB
            inv_timestep = (self.max_num_bb_forecast - 1) - j
            if (len(self.instances[instance_id]) <= inv_timestep):
                continue  # This instance does not have a bb at this timestep
            bb = self.bb_buffer[j][self.instances[instance_id][inv_timestep]]
            bb_orientation = self.get_bb_yaw(bb)
            bb_extent_x = 0.5 * np.sqrt((bb[3, 0] - bb[0, 0]) ** 2 + (bb[3, 1] - bb[0, 1]) ** 2)
            bb_extent_y = 0.5 * np.sqrt((bb[0, 0] - bb[1, 0]) ** 2 + (bb[0, 1] - bb[1, 1]) ** 2)

            speed = np.linalg.norm(bb[5] - bb[4]) #NOTE we are not estimating speed right now so this number is nonsense
            instance_bbs.append(np.array([bb[4,0], bb[4,1], bb_orientation, speed, bb_extent_x, bb_extent_y]))

        return instance_bbs
    

    def update_bb_buffer(self, buffer):
        if(len(self.state_log) < 2): # Start after we have the second measurement
            return
        current_state = np.array(self.state_log[-1])
        last_sate     = np.array(self.state_log[-2])

        R_curr = np.array([[np.cos(current_state[2]), -np.sin(current_state[2])],
                           [np.sin(current_state[2]),  np.cos(current_state[2])]])

        pos_diff =  (R_curr.T @ last_sate[0:2]) - (R_curr.T @ current_state[0:2])

        rot_diff = normalize_angle(last_sate[2] - current_state[2])
        R = np.array([[np.cos(rot_diff), -np.sin(rot_diff)],
                      [np.sin(rot_diff),  np.cos(rot_diff)]])

        for i in range(len(buffer)):  # Loop over timestep
            for j in range(len(buffer[i])): # Loop over Bounding Boxes
                for k in range(buffer[i][j].shape[0]):# Loop over points of the box
                    buffer[i][j][k, :2] = (R @ buffer[i][j][k, :2]) + pos_diff

    def get_bb_yaw(self, box):
        location_2 = box[2]
        location_3 = box[3]
        location_4 = box[4]
        center_top = (0.5 * (location_3 - location_2)) + location_2
        vector_top = center_top - location_4
        rotation_yaw = np.arctan2(vector_top[1], vector_top[0])

        return rotation_yaw

    def prepare_image(self, tick_data):
        image = Image.fromarray(tick_data['rgb'])
        image_degrees = []
        for degree in self.aug_degrees:
            crop_shift = degree / 60 * self.config.img_width
            rgb = torch.from_numpy(self.shift_x_scale_crop(image, scale=self.config.scale, crop=self.config.img_resolution, crop_shift=crop_shift)).unsqueeze(0)
            image_degrees.append(rgb.to('cuda', dtype=torch.float32))
        image = torch.cat(image_degrees, dim=0)
        return image

    def iou_bbs(self, bb1, bb2):
        a = Polygon([(bb1[0,0], bb1[0,1]), (bb1[1,0], bb1[1,1]), (bb1[2,0], bb1[2,1]), (bb1[3,0], bb1[3,1])])
        b = Polygon([(bb2[0,0], bb2[0,1]), (bb2[1,0], bb2[1,1]), (bb2[2,0], bb2[2,1]), (bb2[3,0], bb2[3,1])])
        intersection_area = a.intersection(b).area
        union_area = a.union(b).area
        iou = intersection_area / union_area
        return iou
    
    
    def dot_product(self, vector1, vector2):
        return (vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z)

    def cross_product(self, vector1, vector2):
        return carla.Vector3D(x=vector1.y * vector2.z - vector1.z * vector2.y, y=vector1.z * vector2.x - vector1.x * vector2.z, z=vector1.x * vector2.y - vector1.y * vector2.x)

    def prepare_lidar(self, tick_data):
        lidar_transformed = deepcopy(tick_data['lidar']) 
        lidar_transformed[:, 1] *= -1  # invert
        lidar_transformed = torch.from_numpy(lidar_to_histogram_features(lidar_transformed)).unsqueeze(0)
        lidar_transformed_degrees = [lidar_transformed.to('cuda', dtype=torch.float32)]
        lidar_bev = torch.cat(lidar_transformed_degrees[::-1], dim=1)
        return lidar_bev

    def prepare_goal_location(self, tick_data):
        # tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
        #                                     torch.FloatTensor([tick_data['target_point'][1]])]
        # target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)
        target_point = tick_data['target_point']

        target_point_image_degrees = []
        target_point_degrees = []
        for degree in self.aug_degrees:
            rad = np.deg2rad(degree)
            degree_matrix = np.array([[np.cos(rad), np.sin(rad)],
                                [-np.sin(rad), np.cos(rad)]])

            current_target_point = (degree_matrix @ target_point[0].cpu().numpy().reshape(2, 1)).T

            target_point_image = draw_target_point(current_target_point[0])
            target_point_image = torch.from_numpy(target_point_image)[None].to('cuda', dtype=torch.float32)
            target_point_image_degrees.append(target_point_image)
            target_point_degrees.append(torch.from_numpy(current_target_point))

        target_point_image = torch.cat(target_point_image_degrees, dim=0)
        target_point = torch.cat(target_point_degrees, dim=0).to('cuda', dtype=torch.float32)

        return target_point_image, target_point

    def scale_crop(self, image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
        (width, height) = (image.width // scale, image.height // scale)
        if scale != 1:
            image = image.resize((width, height))
        if crop_x is None:
            crop_x = width
        if crop_y is None:
            crop_y = height
            
        image = np.asarray(image)
        # print(image.shape)
        cropped_image = image[start_y:start_y+crop_y, start_x:start_x+crop_x]
        return cropped_image

    def shift_x_scale_crop(self, image, scale, crop, crop_shift=0):
        crop_h, crop_w = crop
        (width, height) = (int(image.width // scale), int(image.height // scale))
        im_resized = image.resize((width, height))
        image = np.array(im_resized)
        start_y = height//2 - crop_h//2
        start_x = width//2 - crop_w//2
        
        # only shift in x direction
        start_x += int(crop_shift // scale)
        cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
        cropped_image = np.transpose(cropped_image, (2,0,1))
        return cropped_image

    def destroy(self):
        del self.nets

# Taken from LBC
# class RoutePlanner(object):
#     def __init__(self, min_distance, max_distance):
#         self.saved_route = deque()
#         self.route = deque()
#         self.min_distance = min_distance
#         self.max_distance = max_distance
#         self.is_last = False

#         self.mean = np.array([0.0, 0.0]) # for carla 9.10
#         self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10

#     def set_route(self, global_plan, gps=False):
#         self.route.clear()

#         for pos, cmd in global_plan:
#             if gps:
#                 pos = np.array([pos['lat'], pos['lon']])
#                 pos -= self.mean
#                 pos *= self.scale
#             else:
#                 pos = np.array([pos.location.x, pos.location.y])
#                 pos -= self.mean

#             self.route.append((pos, cmd))

#     def run_step(self, gps):
#         if len(self.route) <= 2:
#             self.is_last = True
#             return self.route

#         to_pop = 0
#         farthest_in_range = -np.inf
#         cumulative_distance = 0.0

#         for i in range(1, len(self.route)):
#             if cumulative_distance > self.max_distance:
#                 break

#             cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
#             distance = np.linalg.norm(self.route[i][0] - gps)

#             if distance <= self.min_distance and distance > farthest_in_range:
#                 farthest_in_range = distance
#                 to_pop = i

#         for _ in range(to_pop):
#             if len(self.route) > 2:
#                 self.route.popleft()

#         return self.route

#     def save(self):
#         self.saved_route = deepcopy(self.route)

#     def load(self):
#         self.route = self.saved_route
#         self.is_last = False

def create_BEV(state_log, labels_org, labels_tf, gt_traffic_light_hazard, pred_traffic_light, rgb_image, lidar_bev, target_point, route_buffer, _vehicle, route_buffer_flag,route_gt_map, pix_per_m=5):
    s=0
    max_d = 30
    size = int(max_d*pix_per_m*2)
    origin = (size//2, size//2)
    PIXELS_PER_METER = pix_per_m

    
    # color = [(255, 0, 0), (0, 0, 255)]
    color = [(255), (255)]
   
    
    # create black image
    image_0 = Image.new('L', (size, size))
    image_1 = Image.new('L', (size, size))
    image_2 = Image.new('L', (size, size))
    vel_array = np.zeros((size, size))
    draw0 = ImageDraw.Draw(image_0)
    draw1 = ImageDraw.Draw(image_1)
    draw2 = ImageDraw.Draw(image_2)

    draws = [draw0, draw1, draw2]
    imgs = [image_0, image_1, image_2]
    
    for ix, sequence in enumerate([labels_org, labels_tf]):
               
        # features = rearrange(features, '(vehicle features) -> vehicle features', features=4)
        for ixx, vehicle in enumerate(sequence):
            # draw vehicle
            # if vehicle['class'] != 'Car':
            #     continue
            
            x = -vehicle['position'][1]*PIXELS_PER_METER + origin[1]
            y = -vehicle['position'][0]*PIXELS_PER_METER + origin[0]
            yaw = vehicle['yaw']* 180 / 3.14159265359
            extent_x = vehicle['extent'][2]*PIXELS_PER_METER/2
            extent_y = vehicle['extent'][1]*PIXELS_PER_METER/2
            origin_v = (x, y)
            
            if vehicle['class'] == 'Car':
                p1, p2, p3, p4 = get_coords_BB(x, y, yaw-90, extent_x, extent_y)
                draws[ix].polygon((p1, p2, p3, p4), outline=color[ix]) #, fill=color[ix])
                
                if 'speed' in vehicle:
                    vel = vehicle['speed']*3 #/3.6 # in m/s # just for visu
                    endx1, endy1, endx2, endy2 = get_coords(x, y, yaw-90, vel)
                    draws[ix].line((endx1, endy1, endx2, endy2), fill=color[ix], width=2)

            elif vehicle['class'] == 'Route':
                image = np.array(imgs[ix])
                point = (int(x), int(y))
                cv2.circle(image, point, radius=3, color=color[ix], thickness=3)
                imgs[ix] = Image.fromarray(image)
                
    
    ## fit spline
    if route_buffer_flag and route_gt_map is None:
        x_all = []
        y_all = []
        ego_x = state_log[-1][0]
        ego_y = state_log[-1][1]
        ego_theta = state_log[-1][2]
        rotation_matrix = np.array([[np.cos(ego_theta), -np.sin(ego_theta), 0],
                                    [np.sin(ego_theta), np.cos(ego_theta), 0],
                                    [0, 0, 1]])
        
        ego_pos_global = np.array([ego_x, ego_y, 0])
                    
        for route in route_buffer:
            # ego_vehicle_matrix = np.array(_vehicle.get_transform().get_matrix())
            global_route = route[0] #np.array([route[0].x, route[0].y, route[0].z])
            local_route_pos = global_route - ego_pos_global
            rot = rotation_matrix[:3, :3]
            # local_route_pos = global_route - ego_vehicle_matrix[:3, 3]
            # rot = ego_vehicle_matrix[:3, :3]
            relative_route = rot.T @ local_route_pos
            x = relative_route[1]*PIXELS_PER_METER + origin[1]
            y = -relative_route[0]*PIXELS_PER_METER + origin[0]
            
            if relative_route[0] > -5.0 and np.sqrt(relative_route[1]**2+relative_route[0]**2) < 35:
                x_all.append(y)
                y_all.append(x)
            
                for ix in [0,2]:
                    # ix = 2
                    color_route = [(255), (200), (150), (100)]
                    image = np.array(imgs[ix])
                    point = (int(x), int(y))
                    cv2.circle(image, point, radius=1, color=color_route[0], thickness=1)
                    imgs[ix] = Image.fromarray(image)
            
            

        x_all, y_all = zip(*sorted(zip(x_all, y_all)))
        
        x_all = np.array(x_all)
        y_all = np.array(y_all)
        if x_all.shape[0] == 1:
            # repeat
            x_all = np.concatenate((x_all, x_all))
            y_all = np.concatenate((y_all, y_all))
        s = x_all.shape[0]*100
        if x_all.shape[0] <= 3:
            k = x_all.shape[0]-1
        else:
            k=3
        spline = UnivariateSpline(x_all, y_all, s=s, k=k)
        # x_spline = np.linspace(x_all.min(), x_all.max(), 100)
        x_spline = np.arange(x_all.min(), x_all.max(), 2.5*PIXELS_PER_METER)
        y_spline = spline(x_spline)
        
        # for point in zip(x_spline, y_spline):
        #     pass
        shortened_route = rdp(np.array(list(zip(x_spline, y_spline))), epsilon=0.5*PIXELS_PER_METER)
        ix = 2
        for route in shortened_route:
            color_route = [(255), (200), (150), (100)]
            image = np.array(imgs[ix])
            point = (int(route[1]), int(route[0]))
            cv2.circle(image, point, radius=3, color=color_route[0], thickness=3)
            imgs[ix] = Image.fromarray(image)
            
        
        
        for x,y in zip(x_spline, y_spline):
        # for ix in range(len(imgs)):
            ix = 2
            color_route = [(255), (200), (150), (100)]
            image = np.array(imgs[ix])
            point = (int(y), int(x))
            cv2.circle(image, point, radius=2, color=color_route[0], thickness=2)
            imgs[ix] = Image.fromarray(image)
            
            
          
          
    x = target_point[0][1]*PIXELS_PER_METER + origin[1]
    y = -(target_point[0][0])*PIXELS_PER_METER + origin[0]  
    image = np.array(imgs[0])
    image1 = np.array(imgs[1])
    image2 = np.array(imgs[2])
    point = (int(x), int(y))
    cv2.circle(image, point, radius=2, color=color[0], thickness=2)
    cv2.circle(image1, point, radius=2, color=color[0], thickness=2)
    cv2.circle(image2, point, radius=2, color=color[0], thickness=2)
    imgs[0] = Image.fromarray(image)
    imgs[1] = Image.fromarray(image1)
    imgs[2] = Image.fromarray(image2)
    
    images = [np.asarray(img) for img in imgs]
    image = np.stack([images[0], images[2], images[1]], axis=-1)
    BEV = image

    img_final = Image.fromarray(image.astype(np.uint8))
    if gt_traffic_light_hazard:
        color = 'red'
    else:
        color = 'green'
    img_final = ImageOps.expand(img_final, border=5, fill=color)
    
    if pred_traffic_light:
        color = 'red'
    else:
        color = 'green'
    img_final = ImageOps.expand(img_final, border=5, fill=color)
    
    
    
    ## add rgb image and lidar
    img_final = np.array(img_final)
    rgb_image = rgb_image[0].permute(1, 2, 0).detach().cpu().numpy()[:, :, [2, 1, 0]]
    rgb_image = rgb_image[:,:,::-1]
    images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:2]), axis=1)
    images_lidar = (images_lidar * 255).astype(np.uint8)
    images_lidar = np.stack([images_lidar, images_lidar, images_lidar], axis=-1)

    rgb_image_size = rgb_image.shape[:2]
    images_lidar_size = images_lidar.shape[:2]
    img_final_size = img_final.shape[:2]
    lidar_offset = (img_final_size[0] - images_lidar_size[0], rgb_image_size[1] - images_lidar_size[1] - img_final_size[1])
    images_lidar = np.concatenate([images_lidar, np.zeros_like(images_lidar[:lidar_offset[0]])], axis=0)
    images_lidar = np.concatenate([images_lidar, np.zeros_like(images_lidar[:,:lidar_offset[1]])], axis=1)
    all_images = np.concatenate((images_lidar, img_final), axis=1)
    all_images = np.concatenate((rgb_image, all_images), axis=0)
    all_images = Image.fromarray(all_images.astype(np.uint8))
    
    Path(f'bev_viz2').mkdir(parents=True, exist_ok=True)
    all_images.save(f'bev_viz2/{time.time()}_{s}.png')

    # return BEV

def get_coords(x, y, angle, vel):
    length = vel
    endx2 = x + length * math.cos(math.radians(angle))
    endy2 = y + length * math.sin(math.radians(angle))

    return x, y, endx2, endy2  

def get_coords_BB(x, y, angle, extent_x, extent_y):
    endx1 = x - extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy1 = y + extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx2 = x + extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy2 = y - extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx3 = x + extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy3 = y - extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    endx4 = x - extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy4 = y + extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    return (endx1, endy1), (endx2, endy2), (endx3, endy3), (endx4, endy4)