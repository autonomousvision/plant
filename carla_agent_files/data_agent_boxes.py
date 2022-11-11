import json
import os
import random
import cv2
from copy import deepcopy
from pathlib import Path

import torch
import numpy as np
from rdp import rdp

from carla_agent_files.autopilot import AutoPilot
from scenario_logger import ScenarioLogger
import carla

from carla_agent_files.agent_utils.coordinate_utils import normalize_angle

SHUFFLE_WEATHER = int(os.environ.get('SHUFFLE_WEATHER'))

WEATHERS = {
		'Clear': carla.WeatherParameters.ClearNoon,
		'Cloudy': carla.WeatherParameters.CloudySunset,
		'Wet': carla.WeatherParameters.WetSunset,
		'MidRain': carla.WeatherParameters.MidRainSunset,
		'WetCloudy': carla.WeatherParameters.WetCloudySunset,
		'HardRain': carla.WeatherParameters.HardRainNoon,
		'SoftRain': carla.WeatherParameters.SoftRainSunset,
}

azimuths = [45.0 * i for i in range(8)]

daytimes = {
	'Night': -80.0,
	'Twilight': 0.0,
	'Dawn': 5.0,
	'Sunset': 15.0,
	'Morning': 35.0,
	'Noon': 75.0,
}

WEATHERS_IDS = list(WEATHERS)


def get_entry_point():
    return 'DataAgent'


class DataAgent(AutoPilot):
    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        super().setup(path_to_conf_file, route_index, cfg, exec_or_inter)

        # self.args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cfg = cfg

        self.map_precision = 10.0 # meters per point
        self.rdp_epsilon = 0.5 # epsilon for route shortening

        # radius in which other actors/map elements are considered
        # distance is from the center of the ego-vehicle and measured in 3D space
        self.max_actor_distance = self.detection_radius # copy from expert
        self.max_light_distance = self.light_radius # copy from expert
        self.max_route_distance = 30.0
        self.max_map_element_distance = 30.0

        # if self.log_path is not None:
        #     self.log_path = Path(self.log_path) / route_index
        #     Path(self.log_path).mkdir(parents=True, exist_ok=True) 
        
        # self.scenario_logger = ScenarioLogger(
        #     save_path=self.log_path, 
        #     route_index=self.route_index,
        #     logging_freq=self.save_freq,
        #     log_only=False,
        #     route_only=False, # with vehicles and lights
        #     roi = self.detection_radius+10,
        # )
        
        
        if self.save_path is not None:
            (self.save_path / 'boxes').mkdir()
            
            if self.cfg.SAVE_SENSORS:
                (self.save_path / 'rgb').mkdir()
                (self.save_path / 'rgb_augmented').mkdir()
                (self.save_path / 'lidar').mkdir()
        
    def _init(self, hd_map):
        super()._init(hd_map)

        # if self.scenario_logger:
        #     from srunner.scenariomanager.carla_data_provider import CarlaDataProvider # privileged
        #     self._vehicle = CarlaDataProvider.get_hero_actor()
        #     self.scenario_logger.ego_vehicle = self._vehicle
        #     self.scenario_logger.world = self._vehicle.get_world()

        topology = [x[0] for x in self.world_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        self.polygons = []
        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(self.map_precision)[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(self.map_precision)[0]

            left_marking = [self.lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            right_marking = [self.lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
            self.polygons.append(left_marking + [x for x in reversed(right_marking)])

    def sensors(self):
        result = super().sensors()
        if self.save_path is not None and self.cfg.SAVE_SENSORS:
            result += [{
                'type': 'sensor.camera.rgb',
                'x': self.cfg.camera_pos[0],
                'y': self.cfg.camera_pos[1],
                'z': self.cfg.camera_pos[2],
                'roll': self.cfg.camera_rot_0[0],
                'pitch': self.cfg.camera_rot_0[1],
                'yaw': self.cfg.camera_rot_0[2],
                'width': self.cfg.camera_width,
                'height': self.cfg.camera_height,
                'fov': self.cfg.camera_fov_data_collection,
                'id': 'rgb_front'
            }, {
                'type': 'sensor.camera.rgb',
                'x': self.cfg.camera_pos[0],
                'y': self.cfg.camera_pos[1],
                'z': self.cfg.camera_pos[2],
                'roll': self.cfg.camera_rot_0[0],
                'pitch': self.cfg.camera_rot_0[1],
                'yaw': self.cfg.camera_rot_0[2],
                'width': self.cfg.camera_width,
                'height': self.cfg.camera_height,
                'fov': self.cfg.camera_fov_data_collection,
                'id': 'rgb_augmented'
            }]

            result.append({
                'type': 'sensor.lidar.ray_cast',
                'x': self.cfg.lidar_pos[0],
                'y': self.cfg.lidar_pos[1],
                'z': self.cfg.lidar_pos[2],
                'roll': self.cfg.lidar_rot[0],
                'pitch': self.cfg.lidar_rot[1],
                'yaw': self.cfg.lidar_rot[2],
                'rotation_frequency': self.cfg.lidar_rotation_frequency,
                'points_per_second': self.cfg.lidar_points_per_second,
                'id': 'lidar'
            })


        return result

    def tick(self, input_data):
        result = super().tick(input_data)

        if self.save_path is not None:
            boxes = self.get_bev_boxes()
            
            if self.cfg.SAVE_SENSORS:
                rgb = []
                for pos in ['front']:
                    rgb_cam = 'rgb_' + pos

                    rgb.append(input_data[rgb_cam][1][:, :, :3])

                rgb = np.concatenate(rgb, axis=1)

                rgb_augmented = input_data['rgb_augmented'][1][:, :, :3]

                lidar = input_data['lidar']
            else:
                rgb = None
                rgb_augmented = None
                lidar = None

            
        else:
            rgb = None
            rgb_augmented = None
            boxes = None
            lidar = None


        result.update({'rgb': rgb,
                        'rgb_augmented': rgb_augmented,
                        'boxes': boxes,
                        'lidar': lidar})

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):
        # Must be called before run_step, so that the correct augmentation shift is
        # Saved
        if self.datagen:
            self.augment_camera(sensors)
        control = super().run_step(input_data, timestamp)

        if self.step % self.save_freq == 0:
            if self.save_path is not None:
                tick_data = self.tick(input_data)
                self.save_sensors(tick_data)

            if SHUFFLE_WEATHER and self.step % self.save_freq == 0:
                self.shuffle_weather()
            
            # _, _, _, _ = self.scenario_logger.log_step(self.waypoint_route[:10])
            
        return control


    def augment_camera(self, sensors):
        for sensor in sensors:
            if 'rgb_augmented' in sensor[0]:
                augmentation_translation = np.random.uniform(low=self.cfg.camera_translation_augmentation_min,
                                                            high=self.cfg.camera_translation_augmentation_max)
                augmentation_rotation = np.random.uniform(low=self.cfg.camera_rotation_augmentation_min,
                                                        high=self.cfg.camera_rotation_augmentation_max)
                self.augmentation_translation.append(augmentation_translation)
                self.augmentation_rotation.append(augmentation_rotation)
                camera_pos_augmented = carla.Location(x=self.cfg.camera_pos[0],
                                                    y=self.cfg.camera_pos[1] + augmentation_translation,
                                                    z=self.cfg.camera_pos[2])

                camera_rot_augmented = carla.Rotation(pitch=self.cfg.camera_rot_0[0],
                                                    yaw=self.cfg.camera_rot_0[1] + augmentation_rotation,
                                                    roll=self.cfg.camera_rot_0[2])

                camera_augmented_transform = carla.Transform(camera_pos_augmented, camera_rot_augmented)

                sensor[1].set_transform(camera_augmented_transform)

    def shuffle_weather(self):
        # change weather for visual diversity
        index = random.choice(range(len(WEATHERS)))
        dtime, altitude = random.choice(list(daytimes.items()))
        altitude = np.random.normal(altitude, 10)
        self.weather_id = WEATHERS_IDS[index] + dtime

        weather = WEATHERS[WEATHERS_IDS[index]]
        weather.sun_altitude_angle = altitude
        weather.sun_azimuth_angle = np.random.choice(azimuths)
        self._world.set_weather(weather)

        # night mode
        vehicles = self._world.get_actors().filter('*vehicle*')
        if weather.sun_altitude_angle < 0.0:
            for vehicle in vehicles:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
        else:
            for vehicle in vehicles:
                vehicle.set_light_state(carla.VehicleLightState.NONE)


    def save_sensors(self, tick_data):
        frame = self.step // self.save_freq
        
        if self.cfg.SAVE_SENSORS:
            # CV2 uses BGR internally so we need to swap the image channels before saving.
            cv2.imwrite(str(self.save_path / 'rgb' / (f'{frame:04}.png')), tick_data['rgb'])
            cv2.imwrite(str(self.save_path / 'rgb_augmented' / (f'{frame:04}.png')), tick_data['rgb_augmented'])
            np.save(self.save_path / 'lidar' / ('%04d.npy' % frame), tick_data['lidar'], allow_pickle=True)

        self.save_labels(self.save_path / 'boxes' / ('%04d.json' % frame), tick_data['boxes'])
        
    def save_labels(self, filename, result):
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)
        return

    def save_points(self, filename, points):
        points_to_save = deepcopy(points[1])
        points_to_save[:, 1] = -points_to_save[:, 1]
        np.save(filename, points_to_save)
        return
    
    def destroy(self):
        pass
        # if self.scenario_logger:
        #     self.scenario_logger.dump_to_json()
        #     del self.scenario_logger
    
    def lateral_shift(self, transform, shift):
        transform.rotation.yaw += 90
        transform.location += shift * transform.get_forward_vector()
        return transform

    def get_bev_boxes(self, input_data=None, lidar=None, pos=None):

        # -----------------------------------------------------------
        # Ego vehicle
        # -----------------------------------------------------------

        # add vehicle velocity and brake flag
        ego_location = self._vehicle.get_location()
        ego_transform = self._vehicle.get_transform()
        ego_control   = self._vehicle.get_control()
        ego_velocity  = self._vehicle.get_velocity()
        ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity) # In m/s
        ego_brake = ego_control.brake
        ego_rotation = ego_transform.rotation
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_extent = self._vehicle.bounding_box.extent
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]) * 2.
        ego_yaw =  ego_rotation.yaw/180*np.pi
        relative_yaw = 0
        relative_pos = self.get_relative_transform(ego_matrix, ego_matrix)

        results = []

        # add ego-vehicle to results list
        # the format is category, extent*3, position*3, yaw, points_in_bbox, distance, id
        # the position is in lidar coordinates
        result = {"class": "Car",
                  "extent": [ego_dx[2], ego_dx[0], ego_dx[1] ], #TODO:
                  "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                  "yaw": relative_yaw,
                  "num_points": -1, 
                  "distance": -1, 
                  "speed": ego_speed, 
                  "brake": ego_brake,
                  "id": int(self._vehicle.id),
                }
        results.append(result)
        
        # -----------------------------------------------------------
        # Other vehicles
        # -----------------------------------------------------------

        self._actors = self._world.get_actors()
        vehicles = self._actors.filter('*vehicle*')
        tlights = self._actors.filter('*traffic_light*')
        for vehicle in vehicles:
            if (vehicle.get_location().distance(ego_location) < self.max_actor_distance):
                if (vehicle.id != self._vehicle.id):
                    vehicle_rotation = vehicle.get_transform().rotation
                    vehicle_matrix = np.array(vehicle.get_transform().get_matrix())

                    vehicle_extent = vehicle.bounding_box.extent
                    dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.
                    yaw =  vehicle_rotation.yaw/180*np.pi

                    relative_yaw = normalize_angle(yaw - ego_yaw)
                    relative_pos = self.get_relative_transform(ego_matrix, vehicle_matrix)

                    vehicle_transform = vehicle.get_transform()
                    vehicle_control   = vehicle.get_control()
                    vehicle_velocity  = vehicle.get_velocity()
                    vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity) # In m/s
                    vehicle_brake = vehicle_control.brake

                    # filter bbox that didn't contains points of contains less points
                    if not lidar is None:
                        num_in_bbox_points = self.get_points_in_bbox(ego_matrix, vehicle_matrix, dx, lidar)
                        #print("num points in bbox", num_in_bbox_points)
                    else:
                        num_in_bbox_points = -1

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "Car",
                        "extent": [dx[2], dx[0], dx[1]], #TODO
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "num_points": int(num_in_bbox_points), 
                        "distance": distance, 
                        "speed": vehicle_speed, 
                        "brake": vehicle_brake,
                        "id": int(vehicle.id),
                    }
                    results.append(result)

        # -----------------------------------------------------------
        # Route rdp
        # -----------------------------------------------------------
        if input_data is not None:
            # pos = self._get_position(input_data['gps'][1][:2])
            # self.gps_buffer.append(pos)
            # pos = np.average(self.gps_buffer, axis=0)  # Denoised position
            self._waypoint_planner.load()
            waypoint_route = self._waypoint_planner.run_step(pos)
            self.waypoint_route = np.array([[node[0][0],node[0][1]] for node in waypoint_route])
            self._waypoint_planner.save()
        
        
        max_len = 50
        if len(self.waypoint_route) < max_len:
            max_len = len(self.waypoint_route)
        shortened_route = rdp(self.waypoint_route[:max_len], epsilon=self.rdp_epsilon)
        
        # convert points to vectors
        vectors = shortened_route[1:] - shortened_route[:-1]
        midpoints = shortened_route[:-1] + vectors/2.
        norms = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:,1], vectors[:,0])

        for i, midpoint in enumerate(midpoints):
            # find distance to center of waypoint
            center_bounding_box = carla.Location(midpoint[0], midpoint[1], 0.0)
            transform = carla.Transform(center_bounding_box)
            route_matrix = np.array(transform.get_matrix())
            relative_pos = self.get_relative_transform(ego_matrix, route_matrix)
            distance = np.linalg.norm(relative_pos)
            
            # find distance to beginning of bounding box
            starting_bounding_box = carla.Location(shortened_route[i][0], shortened_route[i][1], 0.0)
            st_transform = carla.Transform(starting_bounding_box)
            st_route_matrix = np.array(st_transform.get_matrix())
            st_relative_pos = self.get_relative_transform(ego_matrix, st_route_matrix)
            st_distance = np.linalg.norm(st_relative_pos)


            # only store route boxes that are near the ego vehicle
            if i > 0 and st_distance > self.max_route_distance:
                continue

            length_bounding_box = carla.Vector3D(norms[i]/2., ego_extent.y, ego_extent.z)
            bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
            bounding_box.rotation = carla.Rotation(pitch = 0.0,
                                                yaw   = angles[i] * 180 / np.pi,
                                                roll  = 0.0)

            route_extent = bounding_box.extent
            dx = np.array([route_extent.x, route_extent.y, route_extent.z]) * 2.
            relative_yaw = normalize_angle(angles[i] - ego_yaw)

            # visualize subsampled route
            # self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1,
            #                             color=carla.Color(0, 255, 255, 255), life_time=(10.0/self.frame_rate_sim))

            result = {
                "class": "Route",
                "extent": [dx[2], dx[0], dx[1]], #TODO
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "centre_distance": distance,
                "starting_distance": st_distance,
                "id": i,
            }
            results.append(result)


        if int(os.environ.get('DATAGEN')):
            # -----------------------------------------------------------
            # Traffic lights
            # -----------------------------------------------------------

            _traffic_lights = self.get_nearby_object(ego_location, tlights, self.max_light_distance)
        
            for light in _traffic_lights:
                if   (light.state == carla.libcarla.TrafficLightState.Red):
                    state = 0
                elif (light.state == carla.libcarla.TrafficLightState.Yellow):
                    state = 1 
                elif (light.state == carla.libcarla.TrafficLightState.Green):
                    state = 2
                else: # unknown
                    state = -1
        
                center_bounding_box = light.get_transform().transform(light.trigger_volume.location)
                center_bounding_box = carla.Location(center_bounding_box.x, center_bounding_box.y, center_bounding_box.z)
                length_bounding_box = carla.Vector3D(light.trigger_volume.extent.x, light.trigger_volume.extent.y, light.trigger_volume.extent.z)
                transform = carla.Transform(center_bounding_box) # can only create a bounding box from a transform.location, not from a location
                bounding_box = carla.BoundingBox(transform.location, length_bounding_box)

                gloabl_rot = light.get_transform().rotation
                bounding_box.rotation = carla.Rotation(pitch = light.trigger_volume.rotation.pitch + gloabl_rot.pitch,
                                                    yaw   = light.trigger_volume.rotation.yaw   + gloabl_rot.yaw,
                                                    roll  = light.trigger_volume.rotation.roll  + gloabl_rot.roll)
                
                light_rotation = transform.rotation
                light_matrix = np.array(transform.get_matrix())

                light_extent = bounding_box.extent
                dx = np.array([light_extent.x, light_extent.y, light_extent.z]) * 2.
                yaw =  light_rotation.yaw/180*np.pi

                relative_yaw = normalize_angle(yaw - ego_yaw)
                relative_pos = self.get_relative_transform(ego_matrix, light_matrix)

                distance = np.linalg.norm(relative_pos)

                result = {
                    "class": "Light",
                    "extent": [dx[2], dx[0], dx[1]], #TODO
                    "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                    "yaw": relative_yaw,
                    "distance": distance, 
                    "state": state, 
                    "id": int(light.id),
                }
                results.append(result)

            # -----------------------------------------------------------
            # Map elements
            # -----------------------------------------------------------

            for lane_id, poly in enumerate(self.polygons):
                for point_id, point in enumerate(poly):
                    if (point.location.distance(ego_location) < self.max_map_element_distance):
                        point_matrix = np.array(point.get_matrix())

                        yaw =  point.rotation.yaw/180*np.pi

                        relative_yaw = yaw - ego_yaw
                        relative_pos = self.get_relative_transform(ego_matrix, point_matrix)
                        distance = np.linalg.norm(relative_pos)

                        result = {
                            "class": "Lane",
                            "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                            "yaw": relative_yaw,
                            "distance": distance,
                            "point_id": int(point_id),
                            "lane_id": int(lane_id),
                        }
                        results.append(result)
                    
        return results

    def get_points_in_bbox(self, ego_matrix, vehicle_matrix, dx, lidar):
        # inverse transform lidar to 
        Tr_lidar_2_ego = self.get_lidar_to_vehicle_transform()
        
        # construct transform from lidar to vehicle
        Tr_lidar_2_vehicle = np.linalg.inv(vehicle_matrix) @ ego_matrix @ Tr_lidar_2_ego

        # transform lidar to vehicle coordinate
        lidar_vehicle = Tr_lidar_2_vehicle[:3, :3] @ lidar[1][:, :3].T + Tr_lidar_2_vehicle[:3, 3:]

        # check points in bbox
        x, y, z = dx / 2.
        # why should we use swap?
        x, y = y, x
        num_points = ((lidar_vehicle[0] < x) & (lidar_vehicle[0] > -x) & 
                      (lidar_vehicle[1] < y) & (lidar_vehicle[1] > -y) & 
                      (lidar_vehicle[2] < z) & (lidar_vehicle[2] > -z)).sum()
        return num_points

    def get_relative_transform(self, ego_matrix, vehicle_matrix):
        """
        return the relative transform from ego_pose to vehicle pose
        """
        relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
        rot = ego_matrix[:3, :3].T
        relative_pos = rot @ relative_pos
        
        # transform to right handed system
        relative_pos[1] = - relative_pos[1]

        # transform relative pos to virtual lidar system
        rot = np.eye(3)
        trans = - np.array([1.3, 0.0, 2.5])
        relative_pos = rot @ relative_pos + trans

        return relative_pos

    def get_lidar_to_vehicle_transform(self):
        # yaw = -90
        rot = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]], dtype=np.float32)
        T = np.eye(4)

        T[0, 3] = 1.3
        T[1, 3] = 0.0
        T[2, 3] = 2.5
        T[:3, :3] = rot
        return T

        
    def get_vehicle_to_lidar_transform(self):
        return np.linalg.inv(self.get_lidar_to_vehicle_transform())

    def get_image_to_vehicle_transform(self):
        # yaw = 0.0 as rot is Identity
        T = np.eye(4)
        T[0, 3] = 1.3
        T[1, 3] = 0.0
        T[2, 3] = 2.3

        # rot is from vehicle to image
        rot = np.array([[0, -1, 0],
                        [0, 0, -1],
                        [1, 0, 0]], dtype=np.float32)
        
        # so we need a transpose here
        T[:3, :3] = rot.T
        return T

    def get_vehicle_to_image_transform(self):
        return np.linalg.inv(self.get_image_to_vehicle_transform())

    def get_lidar_to_image_transform(self):
        Tr_lidar_to_vehicle = self.get_lidar_to_vehicle_transform()
        Tr_image_to_vehicle = self.get_image_to_vehicle_transform()
        T_lidar_to_image = np.linalg.inv(Tr_image_to_vehicle) @ Tr_lidar_to_vehicle
        return T_lidar_to_image