import numpy as np
from collections import deque
import math
from copy import deepcopy



#Filter Functions
def bicycle_model_forward(x, dt, steer, throttle, brake):
    # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
    front_wb = -0.090769015
    rear_wb = 1.4178275

    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    locs_0 = x[0]
    locs_1 = x[1]
    yaw    = x[2]
    speed  = x[3]

    if (brake):
        accel = brake_accel
    else:
        accel = throt_accel * throttle

    wheel = steer_gain * steer

    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = speed + accel * dt
    next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

    next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

    return next_state_x

def measurement_function_hx(vehicle_state):
    '''
    For now we use the same internal state as the measurement state
    :param vehicle_state: VehicleState vehicle state variable containing an internal state of the vehicle from the filter
    :return: np array: describes the vehicle state as numpy array. 0: pos_x, 1: pos_y, 2: rotatoion, 3: speed
    '''
    return vehicle_state

def state_mean(state, Wm):
    '''
    We use the arctan of the average of sin and cos of the angle to calculate the average of orientations.
    :param state: array of states to be averaged. First index is the timestep.
    :param Wm:
    :return:
    '''
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), Wm))
    x[0]    = np.sum(np.dot(state[:, 0], Wm))
    x[1]    = np.sum(np.dot(state[:, 1], Wm))
    x[2]    = math.atan2(sum_sin, sum_cos)
    x[3]    = np.sum(np.dot(state[:, 3], Wm))

    return x

def measurement_mean(state, Wm):
    '''
        We use the arctan of the average of sin and cos of the angle to calculate the average of orientations.
        :param state: array of states to be averaged. First index is the timestep.
        :param Wm:
        :return:
        '''
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), Wm))
    x[0] = np.sum(np.dot(state[:, 0], Wm))
    x[1] = np.sum(np.dot(state[:, 1], Wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], Wm))

    return x

def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x

def normalize_angle_degree(x):
    x = x % 360.0
    if (x > 180.0):
        x -= 360.0
    return x

def residual_state_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y

def residual_measurement_h(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y


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
