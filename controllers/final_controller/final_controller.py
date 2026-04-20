#!/usr/bin/env python3

# Python standard libraries
import json
import imageio.v3 as iio
import math
import numpy as np

# Webots libraries
from controller import Robot

# Path planning libraries
from python_motion_planning.common import *
from python_motion_planning.path_planner import *

# Utility functions
import robot_utils
from robot_utils import (
    Trajectory, PI, 
    world_to_map_coords, convert_path_to_world, convert_diff_drive, 
    getDistance, normalize_angle,
    L, TARGET_VELOCITY, GOAL_TOLERANCE, 
    WHEEL_BASE, WHEEL_RADIUS, MAX_WHEEL_VELOCITY,
    PI_KP, PI_KI, ARM_POSITION_TOLERANCE
)

# -----------------------------
# 1. Webots Robot Setup
# -----------------------------
print("Initializing robot and devices...")
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt_sec = timestep / 1000.0  # dt in seconds for the PI controller

# Robot wheels (actuators)
left_motor = robot.getDevice("wheel_left_joint")
right_motor = robot.getDevice("wheel_right_joint")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Robot localization devices (sensors)
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Initial arm configuration
initial_pose = {
    'torso_lift_joint': 0.3, 'arm_1_joint': 0.71, 'arm_2_joint': 1.02,
    'arm_3_joint': -2.815, 'arm_4_joint': 1.011, 'arm_5_joint': 0,
    'arm_6_joint': 0, 'arm_7_joint': 0,
    'gripper_left_finger_joint': 0.045, 'gripper_right_finger_joint': 0.045,
    'head_1_joint': 0, 'head_2_joint': 0
}
motors = {}
sensors = {}
print("Moving the robot arm to the initial pose...")
for joint_name, position in initial_pose.items():
    try:
        motor = robot.getDevice(joint_name)
        motor.setPosition(position)
        motor.setVelocity(1.0)
        motors[joint_name] = motor
        sensor_name = f"{joint_name}_sensor"
        sensor = robot.getDevice(sensor_name)
        sensor.enable(timestep)
        sensors[joint_name] = sensor
    except Exception as e:
        print(f"Warning: Could not initialize motor/sensor '{joint_name}': {e}")

while robot.step(timestep) != -1:
    all_joints_in_position = True
    for joint_name, sensor in sensors.items():
        current_pos = sensor.getValue()
        target_pos = initial_pose[joint_name]
        if abs(current_pos - target_pos) > ARM_POSITION_TOLERANCE:
            all_joints_in_position = False
            break
    if all_joints_in_position:
        print("Arm is in position. Starting path planning.")
        break

# -----------------------------
# 2. Load Map
# -----------------------------
print("Loading map and metadata...")
with open("occupancy_map.json", "r") as f:
    meta = json.load(f)
image_path = meta["image"]
MAP_RESOLUTION = meta["resolution"]
MAP_ORIGIN = meta["origin"]

img = iio.imread(image_path)
if img.ndim == 3:
    img = np.mean(img, axis=2)
img = np.rot90(img, k=3)
height, width = img.shape
print(f"Map loaded: {width}x{height}px | res={MAP_RESOLUTION} m/px | origin={MAP_ORIGIN}")

# -----------------------------
# 3. Create and Inflate Map
# -----------------------------
#using logic from path planning, inflate radius reduced 0.5 -> 0.3
print("Inflating grid map...")
map_ = Grid(bounds=[[0, width], [0, height]])

map_.type_map[:, :] = TYPES.FREE
map_.type_map[img < 128] = TYPES.OBSTACLE

inflate_radius_m = 0.45
inflate_radius_px = int(np.ceil(inflate_radius_m / MAP_RESOLUTION))
map_.inflate_obstacles(radius=inflate_radius_px)
print(f"Obstacles inflated by {inflate_radius_px} pixels.")

# -----------------------------
# 4. Path Planning
# -----------------------------
# We define the possible targets in the kitchen environment

TARGETS = {
    "green_basket": {"x": -1.24, "y": 0.48, "heading": math.radians(89.02)},
    "nocilla": {"x": 0.69, "y": -0.31, "heading": math.radians(-3.87)},
    "nutella": {"x": 0.28, "y": -1.63, "heading": math.radians(-137.53)},
    "home": {"x": -0.93, "y": -3.14, "heading": math.radians(1.57)}
}

current_target = TARGETS["green_basket"]
GOAL_WORLD = np.array([current_target["x"], current_target["y"]])
TARGET_HEADING = current_target["heading"]

gps_vec = gps.getValues() # Read the current robot location
start_world = np.array([gps_vec[0], gps_vec[1]]) 

start_map = world_to_map_coords(start_world, MAP_ORIGIN, MAP_RESOLUTION)
goal_map = world_to_map_coords(GOAL_WORLD, MAP_ORIGIN, MAP_RESOLUTION)

map_.type_map[start_map] = TYPES.START
map_.type_map[goal_map] = TYPES.GOAL

print("Planning path with AStar...")
planner = AStar(map_=map_, start=start_map, goal=goal_map) #using AStar for faster performance
path, path_info = planner.plan()

if not path: # Stop the robot if no path is found
    print("Could not find a path!")
    path_world = []
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
else:
    print(f"Path found! {len(path)} points.")
    path_world = convert_path_to_world(path, MAP_ORIGIN, MAP_RESOLUTION)
    
    # -----------------------------
    # 5. Initialize Path Display
    # -----------------------------
    static_map_image = None
    
    display = robot.getDevice("map_display")
    map_height, map_width = map_.type_map.shape 
    
    bg_image_data = np.zeros((map_height, map_width, 4), dtype=np.uint8) # Creating an empty image for the display
    bg_image_data[:, :, 0] = bg_image_data[:, :, 1] = bg_image_data[:, :, 2] = img # Filling the image with map data
    bg_image_data[:, :, 3] = 255 

    # Drawing path
    path_map_coords = np.array(path)
    for x, y in path_map_coords:
        if 0 <= x < map_width and 0 <= y < map_height:
            bg_image_data[x, y] = [0, 255, 0, 255] # Green

    # Drawing start and goal
    bg_image_data[start_map[1], start_map[0]] = [0, 0, 255, 255] # Blue
    bg_image_data[goal_map[1], goal_map[0]] = [255, 0, 0, 255] # Red

    # Creating Webots image for the display
    bg_image_flipped = np.rot90(bg_image_data)
    static_map_image = display.imageNew(bg_image_flipped.copy().tobytes(), display.RGBA, map_width, map_height)
    display.imagePaste(static_map_image, 0, 0, blend=False)

    # --- 6. Path Following Loop ---
    print("Starting path following...")
    
    # Setup the trajectory and controller
    traj_x = path_world[:, 0]
    traj_y = path_world[:, 1]
    trajectory = Trajectory(traj_x, traj_y, look_ahead_dist=L)
    PI_angular = PI(dt=dt_sec, kp=PI_KP, ki=PI_KI)

    # Main Control Loop
    plot_counter = 0
    while robot.step(timestep) != -1:
        plot_counter += 1
        
        # Get the robot's current state (Position and Heading)
        gps_vec = gps.getValues()
        compass_vec = compass.getValues()
        curr_pos = [gps_vec[0], gps_vec[1]]
        yaw = math.atan2(compass_vec[0], compass_vec[1])

        # Check if we have reached the goal
        if getDistance(curr_pos, GOAL_WORLD) < GOAL_TOLERANCE:
            print("Goal reached!")
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            break # Exit the path following loop

        # If not, we get the next look-ahead point
        target_point = trajectory.getTargetPoint(curr_pos)

        # Calculate the heading error (Yaw Error)
        target_yaw = math.atan2(target_point[1] - curr_pos[1], target_point[0] - curr_pos[0])
        yaw_error = normalize_angle(target_yaw - yaw)

        # Calculate control signals (Angular velocity)
        target_w = PI_angular.control(yaw_error)
        target_v = TARGET_VELOCITY

        # Convert to wheel velocities (considering velocity limits)
        v_left, v_right = convert_diff_drive(target_v, target_w, WHEEL_BASE, WHEEL_RADIUS)
        max_req_vel = max(abs(v_left), abs(v_right))
        if max_req_vel > MAX_WHEEL_VELOCITY:
            scale = MAX_WHEEL_VELOCITY / max_req_vel
            v_left, v_right = v_left * scale, v_right * scale
        
        # Apply velocities to motors
        left_motor.setVelocity(v_left)
        right_motor.setVelocity(v_right)
        
        # Update map visualization
        if display and plot_counter % 10 == 0: # Update only every 10 steps
            display.imagePaste(static_map_image, 0, 0, blend=False)
            curr_map_pos = world_to_map_coords(curr_pos, MAP_ORIGIN, MAP_RESOLUTION)
            map_x, map_y = curr_map_pos
            display_y = (map_height - 1 - map_y)
            display.setColor(0xFF0000)
            display.fillRectangle(map_x - 1, display_y - 1, 3, 3)
    
    print(f"Position reached! Rotating to final heading: {math.degrees(TARGET_HEADING):.2f}º")
    
    HEADING_TOLERANCE = 0.05 # radians
    
    while robot.step(timestep) != -1:
        # Get current heading
        compass_vec = compass.getValues()
        yaw = math.atan2(compass_vec[0], compass_vec[1])
        
        # Calculate heading error
        yaw_error = normalize_angle(TARGET_HEADING - yaw)
        
        # Check if we are facing the right way
        if abs(yaw_error) < HEADING_TOLERANCE:
            print("Final orientation reached! Ready for manipulation.")
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            break
            
        # Control signals for in-place rotation (Velocity = 0, Angular Velocity depends on error)
        target_w = PI_angular.control(yaw_error) 
        target_v = 0.0 # No forward movement, just rotation
        
        # Convert to wheel velocities
        v_left, v_right = convert_diff_drive(target_v, target_w, WHEEL_BASE, WHEEL_RADIUS)
        
        # Limit velocities
        max_req_vel = max(abs(v_left), abs(v_right))
        if max_req_vel > MAX_WHEEL_VELOCITY:
            scale = MAX_WHEEL_VELOCITY / max_req_vel
            v_left, v_right = v_left * scale, v_right * scale
            
        # Apply velocities
        left_motor.setVelocity(v_left)
        right_motor.setVelocity(v_right)

print("Simulation finished.")