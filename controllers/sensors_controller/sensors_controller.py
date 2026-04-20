#!/usr/bin/env python3

from controller import Robot, Lidar, Keyboard, Motor, GPS, Compass, Camera
import numpy as np

# -----------------------------
# 1. Robot initialization
# -----------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# -----------------------------
# 2. Sensors
# -----------------------------
lidar = robot.getDevice("Hokuyo URG-04LX-UG01")
lidar.enable(timestep)
lidar.enablePointCloud()  # This is to display the point cloud in the 3D simulation

gps = robot.getDevice("gps")
gps.enable(timestep)

compass = robot.getDevice("compass")
compass.enable(timestep)

camera_rgb = robot.getDevice("Astra rgb")
camera_rgb.enable(timestep)

camera_depth = robot.getDevice("Astra depth")
camera_depth.enable(timestep)

# -----------------------------
# 3. Actuators — Differential Drive Wheels
# -----------------------------
left_motor = robot.getDevice("wheel_left_joint")
right_motor = robot.getDevice("wheel_right_joint")

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

max_speed = 3.0  # You can adjust this value as you wish (keeping in mind the robot limits)

# -----------------------------
# 4. Actuators — Arm and Gripper
# -----------------------------
initial_pose = {
    'torso_lift_joint': 0.3, 'arm_1_joint': 0.71, 'arm_2_joint': 1.02,
    'arm_3_joint': -2.815, 'arm_4_joint': 1.011, 'arm_5_joint': 0,
    'arm_6_joint': 0, 'arm_7_joint': 0,
    'gripper_left_finger_joint': 0.045, 'gripper_right_finger_joint': 0.045,
    'head_1_joint': 0, 'head_2_joint': 0
}

motors = {}
for joint_name, position in initial_pose.items():
    try:
        motor = robot.getDevice(joint_name)
        motor.setPosition(position)
        motor.setVelocity(1.0)
        motors[joint_name] = motor
        print(f"Motor '{joint_name}' initialized at position {position}")
    except Exception as e:
        print(f"Warning: Could not initialize motor '{joint_name}': {e}")

# -----------------------------
# 5. Helper Functions
# -----------------------------
def get_heading(compass_values): # Computes robot orientation in degrees from compass readings
    x, y, z = compass_values
    return np.degrees(np.arctan2(x, y))

def set_wheel_speed(left, right): # Sets wheel velocities
    left_motor.setVelocity(left)
    right_motor.setVelocity(right)

# -----------------------------
# 6. Keyboard Control Setup
# -----------------------------
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

print("Controller started successfully.")
print("Use the arrow keys to move the robot.")

# -----------------------------
# 7. Main Loop
# -----------------------------
while robot.step(timestep) != -1:
    # --- Keyboard control ---
    key = keyboard.getKey()
    left_speed = 0.0
    right_speed = 0.0

    if key == Keyboard.UP:
        left_speed = max_speed
        right_speed = max_speed
    elif key == Keyboard.DOWN:
        left_speed = -max_speed
        right_speed = -max_speed
    elif key == Keyboard.LEFT:
        left_speed = -max_speed / 2
        right_speed = max_speed / 2
    elif key == Keyboard.RIGHT:
        left_speed = max_speed / 2
        right_speed = -max_speed / 2

    set_wheel_speed(left_speed, right_speed)

    # --- GPS and Compass info ---
    position = gps.getValues()
    heading = get_heading(compass.getValues())
    print(f"Position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f} | Heading: {heading:.2f}°")

    # --- Sensors (uncomment to print more sensor data) ---
    # lidar_points = lidar.getPointCloud() # Only if point cloud is enabled. If not, the function is lidar.getRangeImage()
    # if lidar_points:
    #     first_point = lidar_points[333]
    #     print(f"LIDAR: {lidar.getNumberOfPoints()} points, coordinates for the middle ray: "
    #       f"X={first_point.x:.3f}, Y={first_point.y:.3f}, Z={first_point.z:.3f} m") # These values are with respect to the lidar reference frame!
    # rgb_image = camera_rgb.getImage()
    # depth_image = camera_depth.getRangeImage()
    # print(f"RGB Camera: {camera_rgb.getWidth()}x{camera_rgb.getHeight()} | Depth: {camera_depth.getWidth()}x{camera_depth.getHeight()}")