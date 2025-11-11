#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIAGo Robot Controller - Pick and Place Operations

Copyright (c) 2025 Panagiotis Georgiadis. All Rights Reserved.

This controller implements a behavior tree architecture for a TIAGo robot
to perform pick and place operations in the Webots simulator.

Author: Panagiotis Georgiadis
Email: pgeorgiadis.it@gmail.com
"""

# ------------------------------------------------------------------------------
# IMPORTS AND CONSTANTS
# ------------------------------------------------------------------------------
import numpy as np
import py_trees
import math
import operator
from ikpy.chain import Chain
from controller import Supervisor
import urdf_parser_py.urdf as urdf_model

# Constants
MAX_MOTOR_SPEED = 6.0
MIN_MOTOR_SPEED = 0.5
ANGLE_TOLERANCE = 0.05  # ~3.5 degrees

# ------------------------------------------------------------------------------
# ROBOT INITIALIZATION AND SETUP
# ------------------------------------------------------------------------------
# Create the Robot instance
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Save and parse URDF file
urdf_path = "Robot.urdf"
with open(urdf_path, "w") as file:
    file.write(robot.getUrdf())
urdf_root = urdf_model.URDF.from_xml_file(urdf_path)

# Parse joint limits
joint_limits = {
    joint.name: {
        "lower": joint.limit.lower,
        "upper": joint.limit.upper,
        "velocity": joint.limit.velocity
    }
    for joint in urdf_root.joint_map.values()
    if joint.limit is not None
}

# Create IK chain
base_elements = [
    "base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint",
    "torso_lift_link", "torso_lift_link_TIAGo front arm_joint", "TIAGo front arm_3",
    "arm_1_joint", "TIAGo front arm_3", "arm_2_joint", "arm_2_link",
    "arm_3_joint", "arm_3_link", "arm_4_joint", "arm_4_link",
    "arm_5_joint", "arm_5_link", "arm_6_joint", "arm_6_link",
    "arm_7_joint", "arm_7_link", "arm_7_link_wrist_ft_tool_link_joint",
    "wrist_ft_tool_link", "wrist_ft_tool_link_front_joint"
]


def create_ik_chain():
    """Create an improved IK chain with better configuration"""
    # First create the chain without an active links mask
    chain = Chain.from_urdf_file(
        urdf_path,
        base_elements=base_elements,
        last_link_vector=[0.016, 0, 0],
        name="tiago_arm"
    )

    # Now create an appropriate active links mask based on the actual chain
    active_links_mask = []
    for i, link in enumerate(chain.links):
        # First link (origin) is always inactive
        if i == 0:
            active_links_mask.append(False)
            continue

        # Explicitly mark fixed links as inactive to avoid warnings
        if hasattr(link, "joint_type") and link.joint_type == "fixed":
            active_links_mask.append(False)
        # Only mark revolute joints as active
        elif hasattr(link, "joint_type") and link.joint_type == "revolute":
            active_links_mask.append(True)
        else:
            active_links_mask.append(False)

    # Create a new chain with the proper active links mask
    print(f"Creating chain with {len(chain.links)} links and mask of length {len(active_links_mask)}")
    return Chain(
        links=chain.links,
        active_links_mask=active_links_mask,
        name="tiago_arm"
    )


# Create the IK chain
ik_chain = create_ik_chain()
# Initialize motors and sensors
part_names = [
    "head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
    "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint",
    "arm_6_joint", "arm_7_joint", "wheel_left_joint", "wheel_right_joint",
    "gripper_left_finger_joint", "gripper_right_finger_joint"
]

motors = {}
sensors = {}

# Define a mapping for special sensor names
special_sensor_names = {
    "gripper_left_finger_joint": "gripper_left_sensor_finger_joint",
    "gripper_right_finger_joint": "gripper_right_sensor_finger_joint"
}

# Initialize all devices
for part_name in part_names:
    try:
        motor = robot.getDevice(part_name)
        limit_info = joint_limits.get(part_name)
        motor.setVelocity(limit_info['velocity'] * 0.3 if limit_info else 1.0)

        # Use special sensor names for gripper fingers, otherwise use default naming
        sensor_name = special_sensor_names.get(part_name, f"{part_name}_sensor")
        sensor = robot.getDevice(sensor_name)

        if sensor:
            sensor.enable(timestep)
            sensors[part_name] = sensor
            print(f"Motor '{part_name}' initialized with sensor '{sensor_name}'")
        else:
            print(f"Warning: Sensor '{sensor_name}' not found for motor '{part_name}'")

        motors[part_name] = motor
    except Exception as e:
        print(f"Failed to initialize motor '{part_name}' or its sensor: {e}")

# Print initialization summary
print("Part names:", part_names)
print(f"Number of motors initialized: {len(motors)}")
print(f"Number of sensors initialized: {len(sensors)}")


def initialize_camera(camera, reduce_resolution=False):
    """Initialize camera with proper resource management"""
    if camera:
        # Reduce resolution if needed
        if reduce_resolution:
            width = camera.getWidth()
            height = camera.getHeight()
            camera.setWidth(max(80, width // 2))
            camera.setHeight(max(60, height // 2))

        # Set up camera correctly
        camera.enable(timestep)
        camera.recognitionEnable(timestep)

        print(f"Camera configured: {camera.getWidth()}x{camera.getHeight()}")


# Enable additional devices
camera = robot.getDevice('camera')
initialize_camera(camera, reduce_resolution=False)

display = robot.getDevice('display')
display.attachCamera(camera)

gps = robot.getDevice('gps')
gps.enable(timestep)

compass = robot.getDevice('compass')
compass.enable(timestep)

# Initialize Lidar
lidar = robot.getDevice("Hokuyo URG-04LX-UG01")
lidar.enable(timestep)

# Enable force feedback for grippers
motors['gripper_left_finger_joint'].enableForceFeedback(timestep)
motors['gripper_right_finger_joint'].enableForceFeedback(timestep)

# Initialize wheel motors
leftMotor = robot.getDevice('wheel_left_joint')
rightMotor = robot.getDevice('wheel_right_joint')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Define predefined positions
starting_position = {
    'torso_lift_joint': 0.3, 'arm_1_joint': 0.71, 'arm_2_joint': 1.02,
    'arm_3_joint': -2.815, 'arm_4_joint': 1.011, 'arm_5_joint': 0,
    'arm_6_joint': 0, 'arm_7_joint': 0, 'gripper_left_finger_joint': 0.045,
    'gripper_right_finger_joint': 0.045, 'head_1_joint': 0, 'head_2_joint': 0
}

# Simplified lift position - based on safe position but higher
lift_position = {
    "torso_lift_joint": 0.3,
    "arm_1_joint": 0.7,
    "arm_2_joint": 0.4,
    "arm_3_joint": -1.5,
    "arm_4_joint": 1.7,
    "arm_5_joint": -1.5,
    "arm_6_joint": 0.0,
    "arm_7_joint": 0.0
}

# Stable place position - controlled extension
place_position = {
    "torso_lift_joint": 0.2,
    "arm_1_joint": 1.6,
    "arm_2_joint": 1.02,
    "arm_3_joint": 0,
    "arm_4_joint": 1.2,
    "arm_5_joint": 0.5,
    "arm_6_joint": 0,
    "arm_7_joint": -2.07,
}

table_waypoints = [
    (1.0, -.9, 0.095),
    (0.2, -1.5, 0.095)  # Then move to table position
]
home_waypoint = [(0.3, 0.0, 0.095)]  # Simple home position

# Set starting position
for joint, position in starting_position.items():
    motors[joint].setPosition(position)


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def camera_to_world_coordinates(camera_position, camera_offset=0.0):
    """
    Converts camera-relative coordinates to global world coordinates using sensor fusion.
    
    Combines GPS, compass, torso height, and camera parameters to calculate precise
    world coordinates of detected objects.
    
    Args:
        camera_position (list): [x, y, z] in camera frame
        camera_offset (float): Z-axis correction offset
        
    Returns:
        list: World coordinates [x, y, z] in meters
    """
    # Get robot's current position and orientation
    robot_pos = gps.getValues()
    compass_val = compass.getValues()
    robot_angle = np.arctan2(compass_val[0], compass_val[1])

    # Calculate rotation matrix
    cos_theta = np.cos(robot_angle)
    sin_theta = np.sin(robot_angle)

    # Forward and right vectors
    forward_x = cos_theta
    forward_y = sin_theta
    right_x = -sin_theta
    right_y = cos_theta

    # Get torso lift height and head tilt
    torso_height = 0
    head_tilt = 0
    if 'torso_lift_joint' in sensors:
        torso_height = sensors['torso_lift_joint'].getValue()
    if 'head_2_joint' in sensors:
        head_tilt = sensors['head_2_joint'].getValue()

    # Calculate precise camera height from URDF measurements
    camera_height = robot_pos[2] + 0.891 + torso_height

    # Camera forward offset from robot center
    camera_forward_offset = 0.25  # Forward offset correction

    # Transform camera coordinates to world coordinates
    world_x = robot_pos[0] + forward_x * (camera_position[0] + camera_forward_offset) + right_x * camera_position[1]
    world_y = robot_pos[1] + forward_y * (camera_position[0] + camera_forward_offset) + right_y * camera_position[1]

    # Apply Z correction based on torso height
    # Reference torso height for normal operation
    reference_torso_height = 0.2

    # Z correction factor based on the observed error:
    # When torso_height = 0.35 (0.15 above reference), Z is overestimated by ~0.28m
    z_correction = 0
    if torso_height > reference_torso_height:
        height_diff = torso_height - reference_torso_height
        z_correction = -1.87 * height_diff  # Empirical correction factor

    # Apply the correction to the Z calculation
    world_z = camera_height + camera_position[2] + z_correction

    # Debug info
    print(f"Robot base height: {robot_pos[2]:.3f}m")
    print(f"Torso lift value: {torso_height:.3f}m")
    print(f"Head tilt: {head_tilt:.3f}rad")
    print(f"Total camera height: {camera_height:.3f}m")
    print(f"Camera-relative Z: {camera_position[2]:.3f}m")
    print(f"Z correction: {z_correction:.3f}m")
    print(f"World Z: {world_z:.3f}m")

    return [world_x, world_y, world_z]


def calculate_approach_offsets(robot_pos, target_pos):
    """Calculate approach vector based on robot-to-target relationship"""
    # Get robot orientation from compass
    compass_val = compass.getValues()
    robot_angle = np.arctan2(compass_val[0], compass_val[1])

    # Calculate vector from robot to target
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)

    # Calculate approach angle (from robot to jar)
    approach_angle = np.arctan2(dy, dx)

    # Calculate approach vector (opposite direction)
    # This creates a vector pointing from the jar toward the robot
    approach_vector_angle = approach_angle + np.pi  # Reverse direction

    # Create approach offset (a small vector pointing from jar toward robot)
    offset_magnitude = 0.045  # 8cm offset, adjust as needed
    offset_x = offset_magnitude * np.cos(approach_vector_angle)
    offset_y = offset_magnitude * np.sin(approach_vector_angle)

    print(f"Robot at ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}), angle {np.degrees(robot_angle):.1f}°")
    print(f"Target at ({target_pos[0]:.3f}, {target_pos[1]:.3f}), distance: {distance:.3f}m")
    print(f"Approach vector: {np.degrees(approach_vector_angle):.1f}°")
    print(f"Calculated offsets: ({offset_x:.3f}, {offset_y:.3f})")

    return offset_x, offset_y


def calculate_inverse_kinematics(target_position, offset_x=0.0, offset_y=0.0):
    """
    Computes joint angles required to position end effector at target location.
    
    Uses IKPY chain with joint limit constraints and initial position clamping
    for reliable solutions.
    
    Args:
        target_position (list): Desired [x, y, z] in world coordinates
        offset_x (float): X-axis safety offset
        offset_y (float): Y-axis safety offset
        
    Returns:
        dict: Joint name to angle mapping or None if no solution
    """

    # Apply offsets directly to the target position
    final_target = [
        target_position[0] + offset_x,
        target_position[1] + offset_y,
        target_position[2]
    ]
    print(f"Attempting IK for final target: {final_target}")

    # Gather and clamp initial joint positions
    initial_position = [
        sensors[joint.name].getValue() if joint.name in sensors else 0.0
        for joint in ik_chain.links
    ]
    for i, link in enumerate(ik_chain.links):
        if link.name in joint_limits:
            lower = joint_limits[link.name]['lower']
            upper = joint_limits[link.name]['upper']
            initial_position[i] = np.clip(initial_position[i], lower, upper)
    print(f"Initial joint positions (clamped): {initial_position}")

    # Perform the IK calculation
    try:
        ik_results = ik_chain.inverse_kinematics(
            target_position=final_target,
            initial_position=initial_position,
            target_orientation=[0, 0, 1],
            orientation_mode="Y"
        )
        print("IK solution found successfully")
        return {
            link.name: ik_results[i]
            for i, link in enumerate(ik_chain.links)
            if link.name in part_names
        }
    except ValueError as e:
        print(f"IK solver error: {e}")
        return None


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return ((angle + math.pi) % (2 * math.pi)) - math.pi


def angle_difference(angle1, angle2):
    """Calculate the difference between two angles"""
    return normalize_angle(angle1 - angle2)


def get_target_position(behavior_name):
    """
    Retrieves target position data from the shared blackboard.
    
    Centralizes position data access across behaviors with consistent
    error handling and logging. Serves as the single source of truth
    for target coordinates throughout the behavior tree.
    
    Args:
        behavior_name (str): Name of the calling behavior (for logging)
    
    Returns:
        List[float]: Target position [x, y, z] or None if not available
    
    Raises:
        No exceptions - errors are caught and logged internally
    """
    try:
        blackboard = py_trees.blackboard.Blackboard()
        target_position = blackboard.get("target_position")
        if target_position is not None:
            print(f"{behavior_name}: Using target position from blackboard: {target_position}")
            return target_position
    except Exception as e:
        print(f"{behavior_name}: Error reading from blackboard: {e}")

    print(f"{behavior_name}: No target position available!")
    return None


def create_movement_with_avoidance(movement_behavior):
    """Wraps a movement behavior with obstacle avoidance"""

    # For testing, simply return the original behavior with no wrapping
    # This will disable obstacle avoidance but keep the behavior tree structure intact
    return movement_behavior

    # The code below would be uncommented when we get lidar working correctly
    '''
    # Create a selector (fallback) node
    movement_selector = py_trees.composites.Selector(
        name=f"Safe_{movement_behavior.name}",
        memory=False  # Re-evaluate children on each tick
    )

    # Add obstacle avoidance with higher priority
    obstacle_avoidance = LidarObstacleAvoidance(
        f"ObstacleAvoidance_{movement_behavior.name}",
        safety_distance=0.5
    )

    # Add behaviors to selector (first one that succeeds is used)
    movement_selector.add_children([obstacle_avoidance, movement_behavior])

    return movement_selector
    '''


# ------------------------------------------------------------------------------
# BEHAVIOR TREE NODE CLASSES
# ------------------------------------------------------------------------------
class LidarObstacleAvoidance(py_trees.behaviour.Behaviour):
    """Uses Lidar to detect and avoid obstacles reactively"""

    def __init__(self, name, safety_distance=0.5, max_speed=3.0):
        super(LidarObstacleAvoidance, self).__init__(name)
        self.safety_distance = safety_distance
        self.max_speed = max_speed
        self.obstacle_detected = False

    def initialise(self):
        self.obstacle_detected = False

    def update(self):
        # Get the Lidar range image
        range_image = lidar.getRangeImage()

        # Calculate sectors for the Lidar readings
        resolution = lidar.getHorizontalResolution()  # Should be 667 based on your screenshot

        # Define the safe field of view (ignore sides where robot body is)
        # For a 240-degree FOV, we'll use the central 180 degrees
        # and ignore the 30 degrees on each side that might see the robot

        # Define sectors for a 240-degree FOV centered at front
        side_ignore_angle = 30  # degrees to ignore on each side
        fov_degrees = 240

        # Calculate how many points to ignore on each side
        points_per_degree = resolution / fov_degrees
        ignore_points = int(side_ignore_angle * points_per_degree)

        # Define the valid range of indices (ignoring the sides)
        valid_start = ignore_points
        valid_end = resolution - ignore_points

        # Now define sectors within the valid range
        sector_width = (valid_end - valid_start) // 5  # 5 sectors across valid FOV

        far_left_start = valid_start
        far_left_end = valid_start + sector_width

        left_start = far_left_end
        left_end = left_start + sector_width

        center_start = left_end
        center_end = center_start + sector_width

        right_start = center_end
        right_end = right_start + sector_width

        far_right_start = right_end
        far_right_end = valid_end

        # Find minimum distances in each sector (with minimum valid range check)
        min_valid_range = 0.25  # Ignore very close readings (likely self-detections)

        center_readings = [r for i, r in enumerate(range_image)
                           if center_start <= i < center_end and r >= min_valid_range and r < 5.0]
        left_readings = [r for i, r in enumerate(range_image)
                         if left_start <= i < left_end and r >= min_valid_range and r < 5.0]
        right_readings = [r for i, r in enumerate(range_image)
                          if right_start <= i < right_end and r >= min_valid_range and r < 5.0]

        # Calculate minimum distances, using infinity if no valid readings
        center_distance = min(center_readings) if center_readings else float('inf')
        left_distance = min(left_readings) if left_readings else float('inf')
        right_distance = min(right_readings) if right_readings else float('inf')

        # Print debugging info occasionally
        if robot.getTime() % 5 < 0.1:  # Every ~5 seconds
            print(
                f"DEBUG - Center dist: {center_distance:.2f}m, Left: {left_distance:.2f}m, Right: {right_distance:.2f}m")
            print(
                f"DEBUG - Valid ranges: {len(center_readings)} center, {len(left_readings)} left, {len(right_readings)} right")

        # Detect obstacles using the filtered readings
        if (center_distance < self.safety_distance or
                left_distance < self.safety_distance * 0.8 or
                right_distance < self.safety_distance * 0.8):

            self.obstacle_detected = True

            # Determine avoidance direction
            if center_distance < self.safety_distance:
                # Choose direction based on which side has more space
                if right_distance > left_distance:
                    print(f"Obstacle ahead! Turning right. Center: {center_distance:.2f}m")
                    leftMotor.setVelocity(self.max_speed * 0.7)
                    rightMotor.setVelocity(-self.max_speed * 0.4)
                else:
                    print(f"Obstacle ahead! Turning left. Center: {center_distance:.2f}m")
                    leftMotor.setVelocity(-self.max_speed * 0.4)
                    rightMotor.setVelocity(self.max_speed * 0.7)
            elif left_distance < self.safety_distance * 0.8:
                print(f"Obstacle on left! Turning right. Left: {left_distance:.2f}m")
                leftMotor.setVelocity(self.max_speed * 0.7)
                rightMotor.setVelocity(-self.max_speed * 0.2)
            elif right_distance < self.safety_distance * 0.8:
                print(f"Obstacle on right! Turning left. Right: {right_distance:.2f}m")
                leftMotor.setVelocity(-self.max_speed * 0.2)
                rightMotor.setVelocity(self.max_speed * 0.7)

            return py_trees.common.Status.RUNNING

        # No obstacles detected
        return py_trees.common.Status.SUCCESS


class EnhancedObjectRecognizer(py_trees.behaviour.Behaviour):
    """
    Computer vision-based object detection and localization behavior.
    
    Features:
    - Multi-sample averaging for position stability
    - Timeout handling
    - Blackboard integration for cross-behavior communication
    - Automatic coordinate conversion to world frame
    
    Args:
        z_offset (float): Vertical offset for grasp positioning
        samples (int): Number of detection samples to average
        timeout (float): Max allowed recognition time
    """

    def __init__(self, name, z_offset=0.0, samples=5, timeout=3.0):
        super(EnhancedObjectRecognizer, self).__init__(name)
        self.z_offset = z_offset
        self.samples = samples
        self.timeout = timeout
        self.target_position = None
        self.object_name = None
        self.start_time = None
        print(f"Created {name} with timeout {self.timeout}s")

    def initialise(self):
        self.target_position = None
        self.object_name = None
        self.start_time = robot.getTime()
        print(f"{self.name}: Initialized recognition with timeout {self.timeout}s")

    def update(self):
        # Safety check for start_time
        if self.start_time is None:
            self.start_time = robot.getTime()

        # Check for timeout
        if (robot.getTime() - self.start_time) > self.timeout:
            print(f"{self.name}: Object recognition timed out after {self.timeout}s")
            return py_trees.common.Status.FAILURE

        object_positions = []
        sample_count = 1 if "After Scan" in self.name else self.samples

        for _ in range(sample_count):
            objects = camera.getRecognitionObjects()
            if objects:
                print(f"{self.name}: Camera sees {len(objects)} objects")

            for obj in objects:
                try:
                    model_name = obj.getModel()
                    if model_name != "jam jar" and model_name != "honey jar":
                        continue

                    # Get position in camera coordinates
                    camera_position = list(obj.getPosition())

                    # Convert to absolute world coordinates using GPS and compass
                    world_position = camera_to_world_coordinates(camera_position, self.z_offset)

                    # Basic filter to eliminate impossible positions
                    if world_position[2] < 0 or world_position[2] > 2.0:
                        continue

                    object_positions.append((world_position, model_name))
                    print(f"{self.name}: Found object: {model_name} at {world_position}")
                except Exception as e:
                    print(f"{self.name}: Error processing recognition object: {e}")

        # If we found objects, choose the closest one
        if object_positions:
            # Get current robot position from GPS
            robot_pos = gps.getValues()

            # Calculate distances to robot and sort
            for i, (pos, _) in enumerate(object_positions):
                dist = sum((pos[j] - robot_pos[j]) ** 2 for j in range(3))
                object_positions[i] = (pos, object_positions[i][1], dist)

            # Sort by distance
            object_positions.sort(key=lambda x: x[2])
            closest_object = object_positions[0]

            self.target_position = closest_object[0]
            self.object_name = closest_object[1]

            # Store results on blackboard without any rotation angle
            try:
                blackboard = py_trees.blackboard.Blackboard()
                blackboard.set("target_position", self.target_position)
                blackboard.set("object_name", self.object_name)
                print(
                    f"{self.name}: Stored target position {self.target_position} and object {self.object_name} on blackboard")
            except Exception as e:
                print(f"{self.name}: Error storing on blackboard: {e}")

            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class ComprehensiveScanner(py_trees.behaviour.Behaviour):
    """
    Systematic environment scanning behavior for object discovery.
    
    Implements:
    - 360-degree rotational scanning
    - Head/camera positioning optimization
    - Progressive scanning with stabilization periods
    - Integrated object recognition at each position
    
    Args:
        total_angles (int): Number of scanning positions
        angle_increment (float): Degrees between positions
        rotation_speed (float): Rad/s for base rotation
    """

    def __init__(self, name, total_angles=8, angle_increment=45, rotation_speed=1.0):
        super(ComprehensiveScanner, self).__init__(name)
        self.total_angles = total_angles
        self.angle_increment = angle_increment
        self.rotation_speed = rotation_speed
        self.current_angle_index = 0
        self.rotation_complete = False
        self.rotation_duration = abs(math.radians(angle_increment) / rotation_speed)

    def initialise(self):
        self.current_angle_index = 0
        self.start_time = robot.getTime()
        self.rotation_complete = False
        print(f"{self.name}: Starting comprehensive scan with {self.total_angles} angles")

        # Setup camera position
        motors['torso_lift_joint'].setPosition(0.35)
        motors['head_1_joint'].setPosition(0.0)
        motors['head_2_joint'].setPosition(-0.2)

    def update(self):
        current_time = robot.getTime()
        time_elapsed = current_time - self.start_time

        # Check if the current rotation is complete
        if self.rotation_complete:
            # Wait for stabilization
            if time_elapsed > self.rotation_duration + 0.3:
                # Check if we've completed all angles
                if self.current_angle_index >= self.total_angles - 1:
                    print(f"{self.name}: Completed full scan")
                    return py_trees.common.Status.SUCCESS

                # Move to next angle
                self.current_angle_index += 1
                self.start_time = current_time
                self.rotation_complete = False
                print(f"{self.name}: Moving to angle {self.current_angle_index + 1}/{self.total_angles}")
                return py_trees.common.Status.RUNNING
            else:
                # Still stabilizing
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                return py_trees.common.Status.RUNNING

        # Check if current rotation is finished
        if time_elapsed >= self.rotation_duration:
            self.rotation_complete = True
            leftMotor.setVelocity(0.0)
            rightMotor.setVelocity(0.0)
            print(f"{self.name}: Rotation {self.current_angle_index + 1}/{self.total_angles} complete")

            # After each rotation, try to recognize
            object_recognizer = EnhancedObjectRecognizer(
                f"Recognize at position {self.current_angle_index + 1}",
                timeout=2.0
            )
            object_recognizer.initialise()
            recognize_status = object_recognizer.update()

            if recognize_status == py_trees.common.Status.SUCCESS:
                print(f"{self.name}: Object recognized after rotation")
                # No need to store rotation angle anymore
                return py_trees.common.Status.SUCCESS

            return py_trees.common.Status.RUNNING

        # Continue rotating
        leftMotor.setVelocity(self.rotation_speed)
        rightMotor.setVelocity(-self.rotation_speed)
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)


class GraspController(py_trees.behaviour.Behaviour):
    """
    Robotic grasping controller with force feedback verification.
    
    State Machine:
    1. APPROACHING: Slowly close gripper until contact
    2. VERIFYING: Maintain grip with force validation
    
    Args:
        force_threshold (float): Minimum grip force (N) for success
    """

    def __init__(self, name, force_threshold=-12.0):
        super(GraspController, self).__init__(name)
        self.force_threshold = force_threshold
        self.state = "APPROACHING"
        self.grip_width = 0.045  # Fully open
        self.verification_time = 0.5
        self.verification_start_time = None
        print(f"{name}: Created simplified grasp controller with threshold={force_threshold}")

    def initialise(self):
        self.state = "APPROACHING"
        self.grip_width = 0.045
        self.verification_start_time = None
        print(f"{self.name}: Starting simplified grasp sequence")

        # Reset blackboard grasp_success
        try:
            blackboard = py_trees.blackboard.Blackboard()
            blackboard.set("grasp_success", False)
        except Exception as e:
            print(f"Error initializing blackboard: {e}")

    def update(self):
        # Get force feedback and position
        left_force = motors['gripper_left_finger_joint'].getForceFeedback()
        right_force = motors['gripper_right_finger_joint'].getForceFeedback()
        current_left = sensors['gripper_left_finger_joint'].getValue()
        current_right = sensors['gripper_right_finger_joint'].getValue()

        # Simple state machine for grasping
        if self.state == "APPROACHING":
            # Move fingers closer slowly
            self.grip_width = max(0.0, self.grip_width - 0.001)
            motors['gripper_left_finger_joint'].setPosition(self.grip_width)
            motors['gripper_right_finger_joint'].setPosition(self.grip_width)

            # Check if both forces indicate contact - SIMPLIFIED CONDITION!
            if abs(left_force) >= abs(self.force_threshold) and abs(right_force) >= abs(self.force_threshold):
                print(f"{self.name}: Contact detected: L={left_force:.2f}, R={right_force:.2f}")
                self.state = "VERIFYING"
                self.verification_start_time = robot.getTime()

        elif self.state == "VERIFYING":
            # Apply a bit more pressure
            target_width = max(0.0, self.grip_width - 0.001)
            motors['gripper_left_finger_joint'].setPosition(target_width)
            motors['gripper_right_finger_joint'].setPosition(target_width)

            current_time = robot.getTime()
            if current_time - self.verification_start_time >= self.verification_time:
                # SUPER SIMPLE VERIFICATION: If both grippers detect force, grasp is good!
                if abs(left_force) >= abs(self.force_threshold) and abs(right_force) >= abs(self.force_threshold):
                    print(f"{self.name}: Grasp successful! L={left_force:.2f}, R={right_force:.2f}")

                    # Set grasp success on blackboard
                    try:
                        blackboard = py_trees.blackboard.Blackboard()
                        blackboard.set("grasp_success", True)
                        print(f"{self.name}: Set grasp_success to True")
                    except Exception as e:
                        print(f"Error setting blackboard: {e}")

                    return py_trees.common.Status.SUCCESS
                else:
                    print(f"{self.name}: Grasp verification failed, retrying")
                    self.state = "APPROACHING"

        # Check for failure conditions - fingers fully closed with no force
        if current_left < 0.005 and current_right < 0.005 and abs(left_force) < abs(self.force_threshold):
            print(f"{self.name}: Grasp failed - fingers closed but no force detected")
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        # Reset blackboard on termination
        try:
            blackboard = py_trees.blackboard.Blackboard()
            blackboard.set("grasp_success", new_status == py_trees.common.Status.SUCCESS)
            print(f"{self.name}: Set grasp_success to {new_status == py_trees.common.Status.SUCCESS}")
        except Exception as e:
            print(f"Error setting blackboard: {e}")

        # If failed, reopen gripper
        if new_status == py_trees.common.Status.FAILURE:
            print(f"{self.name}: Grasp failed - reopening gripper")
            motors['gripper_left_finger_joint'].setPosition(0.045)
            motors['gripper_right_finger_joint'].setPosition(0.045)


# --- Navigation Behaviors ---
class MoveToObject(py_trees.behaviour.Behaviour):
    """
    Navigation behavior that moves robot to target object coordinates.
    
    Implements:
    - GPS and compass-based path planning
    - Dynamic speed control based on distance
    - Obstacle avoidance (using distance sensors)
    - Arm pre-positioning when approaching target
    
    Uses a state machine to handle:
    - INITIAL: Setup and target validation
    - ORIENTING: Rotational alignment to target
    - APPROACHING: Main movement phase
    - STABILIZING: Motion dampening before arm operations
    - ADJUSTING_ARM: Coordinated arm movement during approach
    - FINAL_APPROACH: Precision positioning at target location
    """

    def __init__(self, name, recognize_object, gps, compass, move_arm_behavior, camera):
        super(MoveToObject, self).__init__(name)
        self.current_target = None
        self._recognize_object = recognize_object
        self.gps = gps
        self.compass = compass
        self.move_arm_behavior = move_arm_behavior
        self.camera = camera

        # Distance thresholds
        self.arm_adjustment_distance = 1.28
        self.very_close_distance = 1.0

        # Simplified state machine - removed rotation correction
        self.state = "INITIAL"
        self.start_time = None
        self.stabilization_start_time = None

        # Control parameters
        self.Kp_linear = 0.9
        self.Kp_angular = 1.0
        self.Kd_angular = 1.0
        self.max_speed = 3.0

        # Timing parameters
        self.stabilization_duration = 0.5

    def initialise(self):
        self.state = "INITIAL"
        self.start_time = robot.getTime()

        # Force wheels to stop and stabilize
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)
        robot.step(timestep * 5)

        # Get target position
        self.current_target = get_target_position(self.name)

        if self.current_target is None:
            return py_trees.common.Status.FAILURE

        # Reset motion variables
        self.prev_left_speed = 0.0
        self.prev_right_speed = 0.0
        self.prev_alpha = 0.0
        self.last_time = robot.getTime()

        # Skip to orienting directly - no rotation correction needed with GPS
        self.state = "ORIENTING"

        # Log initial position and target
        robot_pos = self.gps.getValues()
        print(f"{self.name}: Starting from GPS position ({robot_pos[0]:.3f}, {robot_pos[1]:.3f})")
        print(f"{self.name}: Target at ({self.current_target[0]:.3f}, {self.current_target[1]:.3f})")

    def update(self):
        if self.current_target is None:
            return py_trees.common.Status.FAILURE

        # Get current position and orientation from GPS and compass
        current_pos = self.gps.getValues()
        compass_values = self.compass.getValues()
        theta = np.arctan2(compass_values[0], compass_values[1])

        # Calculate distance and angle to target using GPS coordinates
        target_x, target_y = self.current_target[:2]
        dx = target_x - current_pos[0]
        dy = target_y - current_pos[1]
        rho = np.sqrt(dx * dx + dy * dy)
        target_angle = np.arctan2(dy, dx)
        alpha = angle_difference(target_angle, theta)

        # Calculate derivatives for control
        current_time = robot.getTime()
        dt = max(current_time - self.last_time, timestep / 1000.0)
        alpha_rate = (alpha - self.prev_alpha) / dt
        self.prev_alpha = alpha
        self.last_time = current_time

        # Print status occasionally
        if current_time % 1.0 < 0.01:
            print(f"State: {self.state}, Distance: {rho:.4f}m, Angle: {np.degrees(alpha):.2f}°")
            print(f"Robot at ({current_pos[0]:.3f}, {current_pos[1]:.3f}), Target at ({target_x:.3f}, {target_y:.3f})")
            print(f"Robot heading: {np.degrees(theta):.2f}°, Target bearing: {np.degrees(target_angle):.2f}°")

        # STATE MACHINE - simplified without rotation correction
        if self.state == "ORIENTING":
            # Pure orientation phase
            if abs(alpha) < ANGLE_TOLERANCE:
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                print(f"Orientation complete. Angle: {np.degrees(alpha):.2f}°")
                self.state = "APPROACHING"
                return py_trees.common.Status.RUNNING

            # Proportional-derivative control for turning
            turn_speed = 1.0 * alpha - 0.5 * alpha_rate
            turn_speed = np.clip(turn_speed, -1.8, 1.8)
            leftMotor.setVelocity(-turn_speed)
            rightMotor.setVelocity(turn_speed)
            return py_trees.common.Status.RUNNING

        elif self.state == "APPROACHING":
            # Check for arm adjustment
            if rho < self.arm_adjustment_distance:
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                print(f"Stopping to adjust arm at {rho:.2f}m")
                self.stabilization_start_time = current_time
                self.state = "STABILIZING"
                return py_trees.common.Status.RUNNING

            # Keep approaching
            linear_speed = self.Kp_linear * rho
            angular_speed = self.Kp_angular * alpha - self.Kd_angular * alpha_rate

            # Slow down when close
            if rho < 1.5:
                linear_speed *= (0.5 + rho / 3.0)

            # Calculate wheel speeds
            leftSpeed = linear_speed - angular_speed
            rightSpeed = linear_speed + angular_speed

            # Apply limits and set motors
            leftMotor.setVelocity(np.clip(leftSpeed, -self.max_speed, self.max_speed))
            rightMotor.setVelocity(np.clip(rightSpeed, -self.max_speed, self.max_speed))
            return py_trees.common.Status.RUNNING

        elif self.state == "STABILIZING":
            # Wait for robot to fully stop
            if current_time - self.stabilization_start_time < self.stabilization_duration:
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                return py_trees.common.Status.RUNNING

            print("Robot stabilized, starting arm adjustment")
            self.state = "ADJUSTING_ARM"
            if self.move_arm_behavior:
                self.move_arm_behavior.initialise()
            self.arm_adjustment_start_time = current_time
            return py_trees.common.Status.RUNNING

        elif self.state == "ADJUSTING_ARM":
            # Keep motors stopped during arm adjustment
            leftMotor.setVelocity(0.0)
            rightMotor.setVelocity(0.0)

            # Update arm movement
            if self.move_arm_behavior:
                arm_status = self.move_arm_behavior.update()

                # Check if arm movement completed or timed out
                elapsed = current_time - self.arm_adjustment_start_time
                if arm_status == py_trees.common.Status.SUCCESS or elapsed > 2.5:
                    print(f"Arm adjustment complete, continuing approach")

                    # Check if we're already close enough
                    if rho < self.very_close_distance:
                        print(f"Arrived at target, distance: {rho:.2f}m")
                        return py_trees.common.Status.SUCCESS

                    # Otherwise continue approach
                    self.state = "FINAL_APPROACH"
                    return py_trees.common.Status.RUNNING

            return py_trees.common.Status.RUNNING

        elif self.state == "FINAL_APPROACH":
            # Final approach with more careful control
            if rho < self.very_close_distance:
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                print(f"Arrived at target, distance: {rho:.2f}m")
                return py_trees.common.Status.SUCCESS

            # Similar to APPROACHING but more conservative
            linear_speed = 0.5 * self.Kp_linear * rho
            angular_speed = self.Kp_angular * alpha - self.Kd_angular * alpha_rate

            leftSpeed = linear_speed - angular_speed
            rightSpeed = linear_speed + angular_speed

            # Lower max speed for final approach
            max_final_speed = 1.5
            leftMotor.setVelocity(np.clip(leftSpeed, -max_final_speed, max_final_speed))
            rightMotor.setVelocity(np.clip(rightSpeed, -max_final_speed, max_final_speed))
            return py_trees.common.Status.RUNNING

        return py_trees.common.Status.RUNNING


class MoveToWaypoint(py_trees.behaviour.Behaviour):
    """
    Waypoint navigation behavior using a proportional controller approach.

    Features:
    - Simple proportional control for both linear and angular velocity
    - Waypoint sequencing with distance threshold
    - Timeout protection

    Args:
        name (str): Behavior name
        waypoints (list): List of waypoint coordinates [(x1, y1, z1), (x2, y2, z2), ...]
        timeout (float): Maximum allowed execution time
    """

    def __init__(self, name, waypoints, timeout=45.0):
        super(MoveToWaypoint, self).__init__(name)
        self.waypoints = waypoints
        self.timeout = timeout
        self.current_waypoint_index = 0
        self.distance_threshold = 0.15  # Distance to consider waypoint reached

        # Controller parameters from your original function
        self.p1 = 4.0  # Angular gain
        self.p2 = 2.0  # Linear gain
        self.max_speed = 6.28  # Maximum motor speed

        # Timing
        self.start_time = None

    def initialise(self):
        self.current_waypoint_index = 0
        self.start_time = robot.getTime()

        # Stop motors initially
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)

        gps_pos = gps.getValues()[:2]
        print(f"{self.name}: Starting navigation from {gps_pos}")
        print(f"{self.name}: Waypoint sequence: {self.waypoints}")

    def update(self):
        current_time = robot.getTime()

        # Check for timeout
        if current_time - self.start_time > self.timeout:
            print(f"{self.name} timed out after {self.timeout} seconds.")
            leftMotor.setVelocity(0.0)
            rightMotor.setVelocity(0.0)
            return py_trees.common.Status.SUCCESS

        # Get current pose
        xw = gps.getValues()[0]
        yw = gps.getValues()[1]

        # Get compass heading
        compass_vals = compass.getValues()
        theta = np.arctan2(compass_vals[0], compass_vals[1])

        # Current waypoint
        current_waypoint = self.waypoints[self.current_waypoint_index]

        # Calculate error and heading
        rho = np.sqrt((xw - current_waypoint[0]) ** 2 + (yw - current_waypoint[1]) ** 2)
        alpha = np.arctan2(current_waypoint[1] - yw, current_waypoint[0] - xw) - theta

        # Normalize alpha to be between -pi and pi
        if alpha > np.pi:
            alpha = alpha - 2 * np.pi
        elif alpha < -np.pi:
            alpha = alpha + 2 * np.pi

        # Calculate velocity based on error and heading
        vL = -self.p1 * alpha + self.p2 * rho
        vR = +self.p1 * alpha + self.p2 * rho

        # Apply speed limits
        vL = min(vL, self.max_speed)
        vR = min(vR, self.max_speed)
        vL = max(vL, -self.max_speed)
        vR = max(vR, -self.max_speed)

        # Set motor velocities
        leftMotor.setVelocity(vL)
        rightMotor.setVelocity(vR)

        # Log movement occasionally
        if int(current_time) % 3 == 0 and abs(current_time - int(current_time)) < 0.1:
            print(f"GPS=({xw:.2f}, {yw:.2f}), heading={np.degrees(theta):.1f}°, dist to WP={rho:.2f}")

        # Check if waypoint reached
        if rho < self.distance_threshold:
            print(f"Reached waypoint: {current_waypoint} (GPS: {xw:.2f}, {yw:.2f})")

            # Move to next waypoint
            self.current_waypoint_index += 1

            # Check if all waypoints complete
            if self.current_waypoint_index >= len(self.waypoints):
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                print(f"{self.name}: All waypoints reached!")
                return py_trees.common.Status.SUCCESS

            # Continue to next waypoint
            return py_trees.common.Status.RUNNING

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)



# --- Manipulation Behaviors ---
class MoveArmIK(py_trees.behaviour.Behaviour):
    """
    Inverse Kinematics-based arm motion planner.
    
    Implements:
    - Pre-grasp posture positioning
    - Safety offset application
    - Joint limit enforcement
    - Progressive motion validation
    
    Args:
        offset_x (float): X-axis safety margin
        offset_y (float): Y-axis safety margin
        tolerance (float): Joint angle tolerance (rad)
        timeout (float): Max allowed motion time
    """

    def __init__(self, name, offset_x=0.0, offset_y=0.0, tolerance=0.015, timeout=5.0):
        super(MoveArmIK, self).__init__(name)
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.tolerance = tolerance
        self.timeout = timeout

        self.movement_started = False
        self.movement_complete = False
        self.target_angles = None
        self.start_time = None

        # A simple pre-grasp posture (adjust as needed)
        self.pre_grasp_position = {
            "torso_lift_joint": 0.3,
            "arm_1_joint": 0.7,
            "arm_2_joint": 0.4,
            "arm_3_joint": -1.5,
            "arm_4_joint": 1.7,
            "arm_5_joint": -1.5,
            "arm_6_joint": 0.0,
            "arm_7_joint": 0.0
        }

    def initialise(self):
        """Reset internal flags and data each time this behavior starts."""
        self.movement_started = False
        self.movement_complete = False
        self.target_angles = None
        self.start_time = None

    def update(self):
        """Main behavior update cycle."""
        # If we're already done, report SUCCESS
        if self.movement_complete:
            return py_trees.common.Status.SUCCESS

        # 1) Start the IK movement if not yet started
        if not self.movement_started:
            self.start_time = robot.getTime()

            # (a) Get target position (from blackboard or a fallback)
            target_position = get_target_position(self.name)
            if not target_position:
                print(f"{self.name}: No valid target position available.")
                return py_trees.common.Status.FAILURE

            print(f"{self.name}: Target position: {target_position}")

            # (b) Move arm to a pre-grasp posture
            for joint, position in self.pre_grasp_position.items():
                if joint in motors:
                    motors[joint].setPosition(position)

            # Wait a short moment for the arm to settle
            robot.step(timestep * 5)

            # (c) Compute offsets based on approach
            current_robot_pos = gps.getValues()[:2]
            dx, dy = calculate_approach_offsets(current_robot_pos, target_position[:2])

            # (d) Solve IK
            self.target_angles = calculate_inverse_kinematics(
                target_position,
                offset_x=dx + self.offset_x,
                offset_y=dy + self.offset_y
            )
            if not self.target_angles:
                print(f"{self.name}: Failed to calculate IK solution.")
                return py_trees.common.Status.FAILURE

            # (e) Set the joint motors to the IK solution
            for joint, angle in self.target_angles.items():
                if joint in motors:
                    motors[joint].setPosition(angle)

            self.movement_started = True
            print(f"{self.name}: Started arm movement with angles {self.target_angles}")
            return py_trees.common.Status.RUNNING

        # 2) If movement is started, we check progress
        current_time = robot.getTime()
        if (current_time - self.start_time) > self.timeout:
            print(f"{self.name}: Movement timed out—considering it complete.")
            self.movement_complete = True
            return py_trees.common.Status.SUCCESS

        # 3) Check if all joints reached their targets within tolerance
        all_in_place = True
        for joint, target_angle in self.target_angles.items():
            if joint in sensors:
                current_angle = sensors[joint].getValue()
                if abs(target_angle - current_angle) > self.tolerance:
                    all_in_place = False
                    break

        if all_in_place:
            print(f"{self.name}: Arm reached target positions.")
            self.movement_complete = True
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class MoveToPosition(py_trees.behaviour.Behaviour):
    """
    Precise joint-space controller for arm positioning.
    
    Features:
    - Multiple joint coordination with velocity control
    - Position feedback with tolerance checking
    - Progress monitoring to detect stalls/obstructions
    - Auto-completion on timeout with success estimation
    
    Args:
        joint_targets (dict): Mapping of joint names to target positions (rad)
        tolerance (float): Acceptable position error margin (rad)
        timeout (float): Maximum allowed execution time (s)
    """

    def __init__(self, name, joint_targets, tolerance=0.02, timeout=10.0):
        super(MoveToPosition, self).__init__(name)
        self.joint_targets = joint_targets
        self.tolerance = tolerance
        self.movement_complete = False
        self.print_interval = 20
        self.timeout = timeout
        self.start_time = None
        self.progress_time = None
        self.progress_threshold = 3.0  # Consider success if no progress for 3 seconds

    def initialise(self):
        self.movement_complete = False
        self.update_count = 0
        self.start_time = robot.getTime()
        self.progress_time = self.start_time
        self.last_errors = {}

        # Set all joint positions at initialization
        for joint, target in self.joint_targets.items():
            if joint in motors:
                motors[joint].setPosition(target)
                if joint in sensors:
                    self.last_errors[joint] = abs(target - sensors[joint].getValue())
                print(f"Setting {joint} to position {target}")

    def update(self):
        if self.movement_complete:
            return py_trees.common.Status.SUCCESS

        current_time = robot.getTime()

        # Check for timeout
        if current_time - self.start_time > self.timeout:
            print(f"{self.name} timed out after {self.timeout} seconds, considering it complete")
            self.movement_complete = True
            return py_trees.common.Status.SUCCESS

        # Check if we're making progress
        making_progress = False
        all_joints_in_position = True

        for joint, target in self.joint_targets.items():
            if joint not in sensors:
                continue

            current_position = sensors[joint].getValue()
            error = abs(target - current_position)

            # Check if this joint is still moving
            if joint in self.last_errors:
                if abs(self.last_errors[joint] - error) > 0.005:
                    making_progress = True
                self.last_errors[joint] = error

            if error > self.tolerance:
                motors[joint].setPosition(target)
                all_joints_in_position = False

                # Print detailed information for joints that are far from target
                if error > 0.1 and self.update_count % self.print_interval == 0:
                    print(f"Joint {joint} at {current_position:.4f}, target: {target:.4f}, error: {error:.4f}")

        # If we're not making progress for a while, consider it done
        if making_progress:
            self.progress_time = current_time
        elif current_time - self.progress_time > self.progress_threshold:
            print(f"{self.name} stopped making progress, considering it complete")
            self.movement_complete = True
            return py_trees.common.Status.SUCCESS

        self.update_count += 1
        if self.update_count % self.print_interval == 0:
            print(f"{self.name} in progress... ({current_time - self.start_time:.1f}s elapsed)")

        if all_joints_in_position:
            print(f"{self.name} completed successfully.")

            # If this is the lift position after grasping, ensure the object is lifted
            if self.name == "Move to Lift Position":
                # Make sure the arm is raised high enough
                motors['torso_lift_joint'].setPosition(0.35)  # Raise torso
                motors['arm_4_joint'].setPosition(1.5)  # Keep wrist aligned

                # Wait a moment for stability
                robot.step(timestep * 5)

            self.movement_complete = True
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING


class OpenGripper(py_trees.behaviour.Behaviour):
    """
    Controlled gripper opening with position verification.
    
    Carefully opens gripper fingers to release grasped objects while
    monitoring finger positions to ensure complete release.
    
    Implements a timeout mechanism to handle cases where mechanical
    obstructions might prevent full opening.
    
    Args:
        open_position (float): Target finger opening width (m)
        timeout (float): Maximum allowed opening time (s)
    """

    def __init__(self, name, open_position=0.045, timeout=2.0):
        super(OpenGripper, self).__init__(name)
        self.open_position = open_position
        self.timeout = timeout
        self.start_time = None
        self.gripper_opened = False

    def initialise(self):
        self.start_time = robot.getTime()
        self.gripper_opened = False
        print(f"{self.name}: Opening gripper to position {self.open_position}")

        # Set gripper to fully open position
        motors['gripper_left_finger_joint'].setPosition(self.open_position)
        motors['gripper_right_finger_joint'].setPosition(self.open_position)

    def update(self):
        current_time = robot.getTime()

        # Check if we've been running too long
        if current_time - self.start_time > self.timeout:
            print(f"{self.name}: Timeout reached, considering gripper opened")
            return py_trees.common.Status.SUCCESS

        # Check if gripper is fully open
        left_pos = sensors['gripper_left_finger_joint'].getValue()
        right_pos = sensors['gripper_right_finger_joint'].getValue()

        if abs(left_pos - self.open_position) < 0.005 and abs(right_pos - self.open_position) < 0.005:
            if not self.gripper_opened:
                print(f"{self.name}: Gripper fully opened")
                self.gripper_opened = True
                # Wait a bit longer after reaching position
                self.start_time = current_time - self.timeout + 0.5

        # Check if we've waited enough after opening
        if self.gripper_opened and current_time - self.start_time > 0.5:
            print(f"{self.name}: Gripper open sequence complete")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class LiftAndVerify(py_trees.behaviour.Behaviour):
    """
    Object transportation controller with grip integrity monitoring.
    
    Combines:
    - Coordinated joint motion for vertical lifting
    - Continuous force feedback validation
    - Timeout-protected motion execution
    
    Args:
        lift_positions (dict): Target joint angles for lift
        timeout (float): Max lift duration
        force_threshold (float): Minimum required grip force
    """

    def __init__(self, name, lift_positions, timeout=2.0, force_threshold=-5.0):
        super(LiftAndVerify, self).__init__(name)
        self.lift_positions = lift_positions  # Dictionary of joint positions for lifting
        self.timeout = timeout
        self.force_threshold = force_threshold  # Threshold to determine if object is still grasped
        self.start_time = None
        self.movement_started = False

    def initialise(self):
        self.start_time = robot.getTime()
        self.movement_started = False
        print("Starting lift sequence")

        # Stop the robot's wheels
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)

    def update(self):
        current_time = robot.getTime()

        # First check if gripper still has the object
        left_force = motors['gripper_left_finger_joint'].getForceFeedback()
        right_force = motors['gripper_right_finger_joint'].getForceFeedback()

        # If force feedback indicates object was dropped
        if (abs(left_force) < abs(self.force_threshold) and
                abs(right_force) < abs(self.force_threshold)):
            print(f"Object may have been dropped! Forces: L={left_force:.2f}, R={right_force:.2f}")
            # Update blackboard to indicate grasp failure
            blackboard = py_trees.blackboard.Blackboard()
            blackboard.set("grasp_success", False)
            return py_trees.common.Status.FAILURE

        # Start the lift movement if not started yet
        if not self.movement_started:
            # Apply all joint positions from the lift_position dictionary
            for joint_name, position in self.lift_positions.items():
                if joint_name in motors:
                    motors[joint_name].setPosition(position)

            # Ensure gripper maintains its grasp
            # Don't change gripper position as it should maintain the grasp
            self.movement_started = True
            print("Arm moving to lift position")

        # Check for timeout - consider motion complete after timeout
        if current_time - self.start_time > self.timeout:
            print("Lift sequence completed")

            # Final force check to confirm object is still held
            if (abs(left_force) >= abs(self.force_threshold) or
                    abs(right_force) >= abs(self.force_threshold)):
                print(f"Object securely held with forces: L={left_force:.2f}, R={right_force:.2f}")
                return py_trees.common.Status.SUCCESS
            else:
                print("Object lost during final lift position")
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING


# Backup after grasp
class BackupAfterGrasp(py_trees.behaviour.Behaviour):
    """
    Safe backward movement after successful object grasping.
    
    Creates clearance between robot and environment after grasping
    to prevent collisions during subsequent lifting operations.
    
    Features:
    - Controlled linear backward motion
    - Distance-based termination
    - Timeout safety
    - State-based execution (INIT → BACKUP)
    
    Args:
        backup_distance (float): Desired retreat distance (m)
        duration (float): Maximum allowed backup time (s)
    """

    def __init__(self, name, backup_distance=0.12, duration=3.0):
        super(BackupAfterGrasp, self).__init__(name)
        self.backup_distance = backup_distance
        self.duration = duration
        self.start_time = None
        self.start_position = None
        self.state = "INIT"

    def initialise(self):
        self.start_time = robot.getTime()
        self.start_position = gps.getValues()[:2]
        self.state = "INIT"
        print("Starting controlled backup after grasp")

    def update(self):
        current_time = robot.getTime()
        current_position = gps.getValues()[:2]
        dx = current_position[0] - self.start_position[0]
        dy = current_position[1] - self.start_position[1]
        distance_moved = np.sqrt(dx * dx + dy * dy)

        if self.state == "INIT":
            if current_time - self.start_time > 1.0:
                torso_height = sensors['torso_lift_joint'].getValue()
                if torso_height < 0.25:
                    motors['torso_lift_joint'].setPosition(0.30)
                    leftMotor.setVelocity(0)
                    rightMotor.setVelocity(0)
                    return py_trees.common.Status.RUNNING
                else:
                    self.state = "BACKUP"
                    print("Starting backup movement")
            else:
                leftMotor.setVelocity(0)
                rightMotor.setVelocity(0)
                return py_trees.common.Status.RUNNING

        elif self.state == "BACKUP":
            if distance_moved >= self.backup_distance or current_time - self.start_time > self.duration:
                leftMotor.setVelocity(0)
                rightMotor.setVelocity(0)
                print(f"Backup complete, moved {distance_moved:.3f}m")
                return py_trees.common.Status.SUCCESS
            leftMotor.setVelocity(-1.5)
            rightMotor.setVelocity(-1.5)

        return py_trees.common.Status.RUNNING


# --- Decorators and Composites ---
class CheckHardwareStatus(py_trees.behaviour.Behaviour):
    """
    Robot hardware integrity verification behavior.
    
    Systematically checks all required hardware components:
    - Motor controllers
    - Position sensors
    - Camera subsystem
    - Navigation sensors (GPS, compass)
    
    Fails early if any critical component is non-functional,
    preventing unsafe robot operation with faulty hardware.
    """

    def __init__(self, name):
        super(CheckHardwareStatus, self).__init__(name)
        self.required_devices = {
            'motors': ['torso_lift_joint', 'arm_1_joint', 'gripper_left_finger_joint'],
            'sensors': ['torso_lift_joint', 'arm_1_joint', 'gripper_left_finger_joint'],
            'other': ['camera', 'gps', 'compass']
        }

    def update(self):
        # Check motors
        for motor_name in self.required_devices['motors']:
            if motor_name not in motors:
                print(f"ERROR: Required motor '{motor_name}' not found")
                return py_trees.common.Status.FAILURE

        # Check sensors
        for sensor_name in self.required_devices['sensors']:
            if sensor_name not in sensors:
                print(f"ERROR: Required sensor '{sensor_name}' not found")
                return py_trees.common.Status.FAILURE

        # Check other devices
        if not camera:
            print("ERROR: Camera not found")
            return py_trees.common.Status.FAILURE

        if not gps:
            print("ERROR: GPS not found")
            return py_trees.common.Status.FAILURE

        if not compass:
            print("ERROR: Compass not found")
            return py_trees.common.Status.FAILURE

        print("All hardware components verified")
        return py_trees.common.Status.SUCCESS



# ------------------------------------------------------------------------------
# BEHAVIOR TREE CONSTRUCTION
# ------------------------------------------------------------------------------
def create_behavior_tree():
    # Create the root of the behavior tree as a sequence node
    # A sequence executes its children in order and requires all to succeed
    root = py_trees.composites.Sequence(name="Root", memory=True)  # memory=True means it remembers child states
    # between ticks

    # --- INITIALIZATION PHASE ---
    # Create initialization sequence to check hardware and move to a safe starting position
    initialization = py_trees.composites.Sequence(name="Initialization", memory=True)
    check_hardware = CheckHardwareStatus("Check Hardware Status")  # Verifies all required hardware components are
    # working
    move_to_safe_position = MoveToPosition("Move to Safe Position", starting_position)  # Moves arm to predefined
    # safe position
    initialization.add_children([check_hardware, move_to_safe_position])  # Add behaviors to initialization sequence

    # Y-axis offsets for approaching each jar - these are custom values for precise positioning
    # Different offsets are needed for each jar based on their position on the counter
    y_offsets = [0.13, -0.65, -0.6]  # Custom Y offsets for jars 1, 2, and 3

    # Add initialization sequence to the root
    root.add_children([initialization])

    # --- TASK SEQUENCES FOR EACH JAR ---
    # Create a separate handling sequence for each of the 3 jars
    for i in range(3):  # Handle 3 jars
        # Create a sequence for this specific jar
        jar_sequence = py_trees.composites.Sequence(name=f"Handle Jar {i + 1}", memory=True)

        # --- OBJECT DETECTION PHASE ---
        # Create a selector for finding objects (tries first method, falls back to second if needed)
        find_object = py_trees.composites.Selector(name="Find Object", memory=True)  # Selector tries children until
        # one succeeds

        # Try direct recognition first (faster, less movement required)
        recognize = EnhancedObjectRecognizer(f"Recognize Object {i + 1}", timeout=3.0)  # Uses camera to recognize
        # objects

        # If direct recognition fails, use the comprehensive scanner (more thorough but slower)
        comprehensive_scanner = ComprehensiveScanner("Comprehensive Scanner",
                                                    total_angles=8,  # Scan in 8 different directions
                                                    angle_increment=45)  # 45° between scan positions = full 360°
        # coverage

        # Add both detection strategies to the selector
        find_object.add_children([recognize, comprehensive_scanner])

        # --- APPROACH SEQUENCE ---
        # Setup approach sequence to position robot properly before grasping
        approach_sequence = py_trees.composites.Sequence(name="Approach Sequence", memory=True)

        # Prepare arm for approach by moving to pre-approach position
        prepare_arm = MoveToPosition(f"Prepare Arm for Approach {i + 1}", lift_position)  # Moves arm to ready position

        # Create arm movement behavior with custom Y offset for each jar
        # This uses inverse kinematics to position the arm properly for grasping
        move_arm_behavior = MoveArmIK(
            f"Move Arm {i + 1}",
            offset_x=0.0,  # No X-axis offset needed
            offset_y=y_offsets[i]  # Use the custom Y offset for this specific jar
        )

        # Create the move-to-object behavior that navigates the robot to the jar
        basic_move_to_object = MoveToObject(
            f"Move to Object {i + 1}",
            None,  # Target position comes from blackboard, not hardcoded
            gps,  # Provides position information
            compass,  # Provides orientation information
            move_arm_behavior,  # Arm movement to execute during approach
            camera  # Used for visual feedback
        )

        # Wrap navigation with obstacle avoidance capability
        move_to_object = create_movement_with_avoidance(basic_move_to_object)  # Adds LIDAR-based obstacle detection

        # Add behaviors to approach sequence
        approach_sequence.add_children([prepare_arm, move_to_object])

        # --- GRASPING PHASE ---
        # Create grasp behavior that controls the gripper to grab the jar
        grasp_behavior = GraspController(
            f"Grasp Object {i + 1}",
            force_threshold=-10.0  # Force threshold for detecting successful grasp
        )

        # --- TRANSPORT AND PLACEMENT PHASE ---
        # Create sequence for transporting jar to the table and placing it
        transport_and_place = py_trees.composites.Sequence(name="Transport and Place", memory=True)

        # Lift and verify ensures object is securely grasped before transportation
        lift_object = LiftAndVerify(f"Lift and Verify {i + 1}", lift_position)  # Lifts object and checks grip

        # Backup after grasp moves robot backward to create clearance before turning
        backup = BackupAfterGrasp(f"Backup After Grasp {i + 1}")  # Creates space for safe turning

        # Create waypoint navigation behaviors
        basic_move_to_table = MoveToWaypoint(f"Move to Table Waypoint {i + 1}", table_waypoints)  # Navigate to table
        basic_move_to_home = MoveToWaypoint(f"Move to Home Waypoint {i + 1}", home_waypoint)  # Return to home position

        # Wrap waypoint navigation with obstacle avoidance
        move_to_table_bt = create_movement_with_avoidance(basic_move_to_table)
        move_to_home_bt = create_movement_with_avoidance(basic_move_to_home)

        # Create behaviors for placing object on table
        place_object_bt = MoveToPosition(f"Place Object {i + 1}", place_position, timeout=8.0)  # Position arm for
        # placement
        open_gripper_bt = OpenGripper(f"Open Gripper {i + 1}")  # Release the object
        reset_for_home = MoveToPosition(f"Reset Arm For Home {i + 1}", starting_position, timeout=4.0)  # Return arm
        # to safe position

        # Add all transport behaviors to the sequence in proper order
        transport_and_place.add_children([
            py_trees.behaviours.Success(name=f"StartTransport_{i + 1}"),  # Marks start of transport phase
            lift_object,        # 1. Lift and verify secure grasp
            backup,             # 2. Back up for clearance
            move_to_table_bt,   # 3. Navigate to table
            place_object_bt,    # 4. Position arm for placement
            open_gripper_bt,    # 5. Release the object
            reset_for_home,     # 6. Return arm to safe position
            move_to_home_bt     # 7. Return robot to home position
        ])

        # --- COMBINE ALL PHASES ---
        # Add all task phases to the jar sequence in order
        jar_sequence.add_children([
            find_object,        # 1. Find the jar
            approach_sequence,  # 2. Approach the jar
            grasp_behavior,     # 3. Grasp the jar
            transport_and_place # 4. Transport and place the jar
        ])

        # Add this jar's sequence to the root
        root.add_child(jar_sequence)

    # Create the behavior tree from the root
    behavior_tree = py_trees.trees.BehaviourTree(root)
    # Add debug visitor for logging and visualization
    behavior_tree.visitors.append(py_trees.visitors.DebugVisitor())
    return behavior_tree


class RuntimeMonitor:
    """
    System health and performance monitoring subsystem.
    
    Tracks:
    - Joint position histories
    - Navigation trajectories
    - Sensor consistency
    - Battery status (if available)
    - Computational efficiency
    
    Args:
        log_interval (float): Seconds between status reports
    """

    def __init__(self, log_interval=10.0):
        self.start_time = robot.getTime()
        self.last_log_time = self.start_time
        self.log_interval = log_interval
        self.joint_position_history = {}
        self.motor_velocity_history = {}

        # Add GPS position tracking
        self.position_history = []
        self.max_position_history = 100
        self.last_position = None
        if gps:
            self.last_position = gps.getValues()

    def update(self):
        current_time = robot.getTime()

        # Track joint positions
        for joint_name, sensor in sensors.items():
            if joint_name not in self.joint_position_history:
                self.joint_position_history[joint_name] = []
            self.joint_position_history[joint_name].append(sensor.getValue())
            # Limit history length
            if len(self.joint_position_history[joint_name]) > 100:
                self.joint_position_history[joint_name].pop(0)

        # Track GPS position
        if gps:
            current_pos = gps.getValues()
            self.position_history.append(current_pos)
            if len(self.position_history) > self.max_position_history:
                self.position_history.pop(0)

            # Calculate movement speed
            if self.last_position:
                dx = current_pos[0] - self.last_position[0]
                dy = current_pos[1] - self.last_position[1]
                distance = np.sqrt(dx * dx + dy * dy)
                dt = timestep / 1000.0
                speed = distance / dt if dt > 0 else 0
                self.last_position = current_pos

        # Log periodically
        if current_time - self.last_log_time >= self.log_interval:
            self.log_robot_status()
            self.last_log_time = current_time

    def log_robot_status(self):
        """Log important robot metrics with enhanced GPS and compass data"""
        # Robot position and orientation
        if gps and compass:
            pos = gps.getValues()
            compass_vals = compass.getValues()
            heading = np.degrees(np.arctan2(compass_vals[0], compass_vals[1]))

            print(f"GPS Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            print(f"Heading: {heading:.2f}° ({compass_vals[0]:.2f}, {compass_vals[1]:.2f})")

            # Calculate path stats if we have enough history
            if len(self.position_history) > 2:
                # Calculate total distance traveled
                total_distance = 0
                for i in range(1, len(self.position_history)):
                    p1 = self.position_history[i - 1]
                    p2 = self.position_history[i]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    total_distance += np.sqrt(dx * dx + dy * dy)

                # Calculate displacement (straight-line distance from start)
                start = self.position_history[0]
                current = self.position_history[-1]
                dx = current[0] - start[0]
                dy = current[1] - start[1]
                displacement = np.sqrt(dx * dx + dy * dy)

                print(f"Total distance traveled: {total_distance:.2f}m")
                print(f"Displacement from start: {displacement:.2f}m")
                if total_distance > 0:
                    print(f"Path efficiency: {(displacement / total_distance) * 100:.1f}%")

        # Battery status (if available)
        battery_level = robot.getBatteryValue() if hasattr(robot, "getBatteryValue") else "N/A"
        print(f"Battery level: {battery_level}")

        # Check for any excessive joint velocities or oscillations
        for joint_name, history in self.joint_position_history.items():
            if len(history) > 20:
                # Check for oscillations by looking at position changes
                changes = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
                if sum(changes) > 0.5 and max(changes) < 0.1:
                    print(f"WARNING: Possible oscillation in joint {joint_name}")


# ------------------------------------------------------------------------------
# MAIN EXECUTION LOOP
# ------------------------------------------------------------------------------
def main():
    """Main execution function"""
    # Verify GPS and compass are working
    if gps and compass:
        pos = gps.getValues()
        compass_vals = compass.getValues()
        heading = np.degrees(np.arctan2(compass_vals[0], compass_vals[1]))
        print(f"Initial GPS Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"Initial Heading: {heading:.2f}°")
    else:
        print("WARNING: GPS or compass unavailable!")

    # Create runtime monitor
    monitor = RuntimeMonitor(log_interval=10.0)

    # Use the behavior tree to control the robot
    behavior_tree = create_behavior_tree()
    behavior_tree.setup(timeout=15)

    # Initialize the tree with a first tick to get proper status
    behavior_tree.tick()

    last_print_time = robot.getTime()
    print_interval = 10.0  # Seconds between status prints
    tick_interval = 1  # Ticks per timestep

    while robot.step(timestep) != -1:
        if robot.getBasicTimeStep() % tick_interval == 0:
            # Update runtime monitor
            monitor.update()

            # Tick the behavior tree
            behavior_tree.tick()
            current_time = robot.getTime()

            # Get the status from the root node (not the tick() return value)
            status = behavior_tree.root.status

            if current_time - last_print_time >= print_interval:
                print("\nBehavior Tree Status:")
                print(py_trees.display.ascii_tree(behavior_tree.root))
                last_print_time = current_time

            if status in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
                print(
                    f"Behavior tree {'completed successfully' if status == py_trees.common.Status.SUCCESS else 'failed'}")
                break

        robot.step(timestep)


# Run the main function
if __name__ == "__main__":
    main()
