import math
import numpy as np

# =================================================================
# --- Constants and Parameters ---
# =================================================================
# --- TIAGo Parameters  ---
WHEEL_BASE = 0.4044      # Distance between the wheels (m)
WHEEL_RADIUS = 0.0985    # Wheel radius (m)
MAX_WHEEL_VELOCITY = 10.1523 # Max wheel speed (rad/s)

# --- Arm/Gripper Constants ---
ARM_POSITION_TOLERANCE = 0.01 # Radian tolerance

# --- Path Following Constants ---
L = 0.5 # Look-Ahead Distance (meters)
TARGET_VELOCITY = 0.2 # Target linear velocity (m/s)
GOAL_TOLERANCE = 0.1 # Goal tolerance (m)

# --- PI Controller Gains (these gains control angular velocity) ---
PI_KP = 0.5  # Proportional gain (reactivity)
PI_KI = 0.05 # Integral gain (corrects small, steady errors)

# =================================================================
# --- Navigation and Distance Functions ---
# =================================================================
# --- Distance between two 2D points  ---
def getDistance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)

# --- Normalize an angle to the range [-pi, pi] ---
def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

# =================================================================
# --- Pure Pursuit Functions and Classes ---
# =================================================================
class Trajectory:
    # Holds the reference path and finds the look-ahead point for the controller
    def __init__(self, traj_x, traj_y, look_ahead_dist):
        self.traj_x = traj_x
        self.traj_y = traj_y
        self.last_idx = 0
        self.L = look_ahead_dist # Look-ahead distance

    def getPoint(self, idx):
        return [self.traj_x[idx], self.traj_y[idx]]

    def getTargetPoint(self, pos):
        # Get the next look ahead point
        target_idx = self.last_idx
        target_point = self.getPoint(target_idx)
        curr_dist = getDistance(pos, target_point)

        # Loop to find the closest point that is beyond L
        while curr_dist < self.L and target_idx < len(self.traj_x) - 1:
            target_idx += 1
            target_point = self.getPoint(target_idx)
            curr_dist = getDistance(pos, target_point)

        # Try to find a point that is closer to L
        if target_idx > 0:
            prev_point = self.getPoint(target_idx - 1)
            prev_dist = getDistance(pos, prev_point)
            if prev_dist > curr_dist: # If the previous point is closer to L
                 target_idx = target_idx - 1

        self.last_idx = target_idx
        return self.getPoint(target_idx)

class PI:
    # Simple Proportional-Integral (PI) controller to calculate the angular velocity
    def __init__(self, dt, kp=1.0, ki=0.1):
        self.kp = kp # Proportional Gain (if too high, the robot oscillates; if too low, the robot corrects its pose too slow)
        self.ki = ki # Integral Gain (if too high, it may cause instability; it too low the robot may never reach the target)
        
        self.dt = dt # Delta time (time since last update)
        self.Pterm = 0.0 # Proportional term
        self.Iterm = 0.0 # Integral term
        self.last_error = 0.0

    def control(self, error):
        # Calculates the control output given a new error value.
        self.Pterm = self.kp * error
        self.Iterm += error * self.dt # Accumulate error over time

        self.last_error = error
        output = self.Pterm + (self.ki * self.Iterm)
        
        return output

# =================================================================
# --- Coordinate and Velocity Transform Functions ---
# =================================================================
def world_to_map_coords(world_pos, origin, resolution):
    map_x = int((world_pos[0] - origin[0]) / resolution)
    map_y = int((world_pos[1] - origin[1]) / resolution)
    return (map_x, map_y)

def convert_path_to_world(path_map, origin, resolution):
    path_world = []
    for point in path_map:
        world_x = (point[0] * resolution) + origin[0]
        world_y = (point[1] * resolution) + origin[1]
        world_x += resolution / 2.0
        world_y += resolution / 2.0
        path_world.append([world_x, world_y])
    return np.array(path_world)

def convert_diff_drive(v, w, wheel_base, wheel_radius):
    # Converts target linear (v) and angular (w) velocities into left and right wheel speeds (in rad/s)
    v_left = (v - (w * wheel_base / 2.0)) / wheel_radius
    v_right = (v + (w * wheel_base / 2.0)) / wheel_radius
    return v_left, v_right