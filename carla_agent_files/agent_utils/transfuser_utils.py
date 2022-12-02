"""
Some utility functions e.g. for normalizing angles
Functions for detecting red lights are adapted from scenario runners
atomic_criteria.py
"""
import math
# import carla
import numpy as np
import torch
import cv2
from collections import deque


def normalize_angle(x):
  x = x % (2 * np.pi)  # force in range [0, 2 pi)
  if x > np.pi:  # move to [-pi, pi)
    x -= 2 * np.pi
  return x


def normalize_angle_degree(x):
  x = x % 360.0
  if x > 180.0:
    x -= 360.0
  return x


# def rotate_point(point, angle):
#   """
#   rotate a given point by a given angle
#   """
#   x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
#   y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
#   return carla.Vector3D(x_, y_, point.z)


# def get_traffic_light_waypoints(traffic_light, carla_map):
#   """
#   get area of a given traffic light
#   """
#   base_transform = traffic_light.get_transform()
#   base_loc = traffic_light.get_location()
#   base_rot = base_transform.rotation.yaw
#   area_loc = base_transform.transform(traffic_light.trigger_volume.location)

#   # Discretize the trigger box into points
#   area_ext = traffic_light.trigger_volume.extent
#   x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes

#   area = []
#   for x in x_values:
#     point = rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
#     point_location = area_loc + carla.Location(x=point.x, y=point.y)
#     area.append(point_location)

#   # Get the waypoints of these points, removing duplicates
#   ini_wps = []
#   for pt in area:
#     wpx = carla_map.get_waypoint(pt)
#     # As x_values are arranged in order, only the last one has to be checked
#     if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
#       ini_wps.append(wpx)

#   # Advance them until the intersection
#   wps = []
#   eu_wps = []
#   for wpx in ini_wps:
#     distance_to_light = base_loc.distance(wpx.transform.location)
#     eu_wps.append(wpx)
#     next_distance_to_light = distance_to_light + 1.0
#     while not wpx.is_intersection:
#       next_wp = wpx.next(0.5)[0]
#       next_distance_to_light = base_loc.distance(next_wp.transform.location)
#       if next_wp and not next_wp.is_intersection \
#           and next_distance_to_light <= distance_to_light:
#         eu_wps.append(next_wp)
#         distance_to_light = next_distance_to_light
#         wpx = next_wp
#       else:
#         break

#     if not next_distance_to_light <= distance_to_light and len(eu_wps) >= 4:
#       wps.append(eu_wps[-4])
#     else:
#       wps.append(wpx)

#   return area_loc, wps


def logit_norm(x, t=0.05, dim=1):
  norms = torch.norm(x, p=2, dim=dim, keepdim=True) + 1e-7
  logit_norm_local = torch.div(x, norms) / t
  return logit_norm_local


def lidar_to_ego_coordinate(config, lidar):
  """
  Converts the LiDAR points given by the simulator into the ego agents
  coordinate system
  :param config: GlobalConfig, used to read out lidar orientation and location
  :param lidar: the LiDAR point cloud as provided in the input of run_step
  :return: lidar where the points are w.r.t. 0/0/0 of the car and the carla
  coordinate system.
  """
  yaw = np.deg2rad(config.lidar_rot[2])
  rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

  translation = np.array(config.lidar_pos)

  # The double transpose is a trick to compute all the points together.
  ego_lidar = (rotation_matrix @ lidar[1][:, :3].T).T + translation

  return ego_lidar


def algin_lidar(lidar, translation, yaw):
    """
    Translates and rotates a LiDAR into a new coordinate system.
    Rotation is inverse to translation and yaw
    :param lidar: numpy LiDAR point cloud (N,3)
    :param translation: translations in meters
    :param yaw: yaw angle in radians
    :return: numpy LiDAR point cloud in the new coordinate system.
    """

    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

    aligned_lidar = (rotation_matrix.T @ (lidar - translation).T).T

    return aligned_lidar


def inverse_conversion_2d(point, translation, yaw):
  """
  Performs a forward coordinate conversion on a 2D point
  :param point: Point to be converted
  :param translation: 2D translation vector of the new coordinate system
  :param yaw: yaw in radian of the new coordinate system
  :return: Converted point
  """
  rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

  converted_point = rotation_matrix.T @ (point - translation)
  return converted_point


def preprocess_compass(compass):
  """
  Checks the compass for Nans and rotates it into the default CARLA coordinate
  system with range [-pi,pi].
  :param compass: compass value provided by the IMU, in radian
  :return: yaw of the car in radian in the CARLA coordinate system.
  """
  if math.isnan(compass):  # simulation bug
    compass = 0.0
  # The minus 90.0 degree is because the compass sensor uses a different
  # coordinate system then CARLA. Check the coordinate_sytems.txt file
  compass = normalize_angle(compass - np.deg2rad(90.0))

  return compass


def get_relative_transform(ego_matrix, vehicle_matrix):
  """
  Returns the position of the vehicle matrix in the ego coordinate system.
  :param ego_matrix: ndarray 4x4 Matrix of the ego vehicle in global
  coordinates
  :param vehicle_matrix: ndarray 4x4 Matrix of another actor in global
  coordinates
  :return: ndarray position of the other vehicle in the ego coordinate system
  """
  relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
  rot = ego_matrix[:3, :3].T
  relative_pos = rot @ relative_pos

  return relative_pos


def draw_point_bev(bev_images, point, color=(255, 255, 255), thickness=2, radius=2):
  for idx, bev_img in enumerate(bev_images):
    cv2.circle(bev_img, point, radius=radius, color=color[idx], thickness=thickness)


class PIDController(object):
  """
    PID controller that converts waypoints to steer, brake and throttle commands
    """

  def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, n=20):
    self.k_p = k_p
    self.k_i = k_i
    self.k_d = k_d

    self.window = deque([0 for _ in range(n)], maxlen=n)

  def step(self, error):
    self.window.append(error)

    if len(self.window) >= 2:
      integral = np.mean(self.window)
      derivative = (self.window[-1] - self.window[-2])
    else:
      integral = 0.0
      derivative = 0.0

    return self.k_p * error + self.k_i * integral + self.k_d * derivative


def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
  """ Adapted from mmdetection
  Args:
      pred (torch.Tensor): The prediction.
      gaussian_target (torch.Tensor): The learning target of the prediction
          in gaussian distribution.
      alpha (float, optional): A balanced form for Focal Loss.
          Defaults to 2.0.
      gamma (float, optional): The gamma for calculating the modulating
          factor. Defaults to 4.0.
  """
  eps = 1e-12
  pos_weights = gaussian_target.eq(1)
  neg_weights = (1 - gaussian_target).pow(gamma)
  pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
  neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
  return pos_loss + neg_loss
