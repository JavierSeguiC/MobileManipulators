import py_trees
import numpy as np
from robot import TiagoRobot
import math
import json
import imageio.v3 as iio
from python_motion_planning.common import *
from python_motion_planning.path_planner import *
from robot_utils import (
    Trajectory, PI, world_to_map_coords, convert_path_to_world, 
    convert_diff_drive, getDistance, normalize_angle,
    L, TARGET_VELOCITY, GOAL_TOLERANCE, WHEEL_BASE, WHEEL_RADIUS, 
    MAX_WHEEL_VELOCITY, PI_KP, PI_KI, ARM_POSITION_TOLERANCE
)

 


class MoveArmTrajectoryRRT(py_trees.behaviour.Behaviour):
    """
    Generic Behavior Tree node to move the arm using RRT.
    Executes a sequence of movements based on the provided list of offsets.
    """

    def __init__(self, name, robot, offsets, use_target_from_blackboard=False, fixed_target=None, tolerance=0.02, timeout=200.0):
        super(MoveArmTrajectoryRRT, self).__init__(name)
        self.robot : TiagoRobot = robot
        self.offsets = offsets
        self.use_target_from_blackboard = use_target_from_blackboard
        self.fixed_target = fixed_target
        self.tolerance = tolerance
        self.timeout = timeout
        
        self.movement_complete = False
        self.start_time = None
        self.current_phase = 0
        self.blackboard = py_trees.blackboard.Blackboard()

    def initialise(self):
        # Reset variables for a new execution
        self.movement_complete = False
        self.start_time = self.robot.get_time()
        self.current_phase = 0
            
        print(f"{self.name}: Initializing RRT trajectory planning...")

    def update(self):
        if self.movement_complete:
            return py_trees.common.Status.SUCCESS

        current_time = self.robot.get_time()

        if current_time - self.start_time > self.timeout:
            print(f"{self.name}: Timed out after {self.timeout} seconds.")
            return py_trees.common.Status.FAILURE

        # --- FASE RRT ---
        base_target = None
        if self.use_target_from_blackboard:
            base_target = self.blackboard.get("target_position")
            if base_target is None:
                print(f"{self.name}: Waiting for \"target_position\" on blackboard...")
                return py_trees.common.Status.RUNNING
        elif self.fixed_target is not None:
            base_target = self.fixed_target
        else:
            print(f"{self.name}: ERROR - No target provided.")
            return py_trees.common.Status.FAILURE

        base_target = np.array(base_target)

        current_offset = np.array(self.offsets[self.current_phase])
        goal_pos = base_target + current_offset
        
        current_joint_positions = self.robot.read_torso_and_arm_joints()
        current_end_effector_pos = self.robot.ik_chain.calculate_forward_kinematics(current_joint_positions)
        
        print(f"{self.name}: Planning RRT to phase {self.current_phase + 1}/{len(self.offsets)} (Goal: {goal_pos})")
        path = self.robot.planner.run_rrt_with_obstacles(
            x_init = tuple(current_end_effector_pos), 
            x_goal = tuple(goal_pos),
            headless = True
        )

        if not path:
            print(f"{self.name}: RRT Planner failed to find a path.")
            return py_trees.common.Status.FAILURE

        target_angles = {}
        for waypoint in path:
            target_angles = self.robot.ik_chain.calculate_inverse_kinematics(
                waypoint, [0, 0, 1], orientation_mode="Z"
            )
            if not target_angles:
                print(f"{self.name}: Failed to calculate IK solution for waypoint.")
                return py_trees.common.Status.FAILURE

            for joint, angle in target_angles.items():
                self.robot.set_joint_position(joint, angle)
            
            self.robot.step(200)  
            
        all_in_place = True
        for joint, target_angle in target_angles.items():
            current_angle = self.robot.get_joint_position(joint)
            
            if abs(target_angle - current_angle) > self.tolerance:
                all_in_place = False
                break

        if all_in_place:
            print(f"{self.name}: Reached phase {self.current_phase + 1} target.")
            self.current_phase += 1
            
            if self.current_phase >= len(self.offsets):
                self.movement_complete = True
                return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING
    

class EnhancedObjectRecognizer(py_trees.behaviour.Behaviour):
    """
    Computer vision-based object detection and localization behavior.
    """

    def __init__(self, name, robot_instance: TiagoRobot, z_offset=0.0, samples=5, timeout=30.0):
        super(EnhancedObjectRecognizer, self).__init__(name)
        self.robot = robot_instance  
        self.z_offset = z_offset
        self.samples = samples
        self.timeout = timeout
        self.target_position = None
        self.object_name = None
        self.start_time = None
        self.joint_name = 'head_1_joint'
        self.waypoints = [1.0, -1.0, 0.0] 
        self.current_idx = 0
        self.tolerance = 0.05
        self.speed = 1.0
        
        self.object_positions = [] 
        
        self.blackboard = py_trees.blackboard.Blackboard()
        print(f"Created {name} with timeout {self.timeout}s")

    def initialise(self):
        self.target_position = None
        self.object_name = None
        self.start_time = self.robot.get_time()
        self.current_idx = 0
        self.object_positions = [] 
        
        print(f"{self.name}: Iniciando barrido de cabeza...")
        
        # Initial head position and speed setup using the robot instance
        self.robot.set_joint_position('torso_lift_joint', 0.0)
        self.robot.set_joint_position('head_2_joint', -0.5) 
        self.robot.step(50) # Allow time for head to move

        if self.joint_name in self.robot.motors:
            self.robot.motors[self.joint_name].setVelocity(self.speed)
        
        self.set_target()
        print(f"{self.name}: Initialized recognition with timeout {self.timeout}s")
        
    def set_target(self):
        if self.current_idx < len(self.waypoints):
            target = self.waypoints[self.current_idx]
            self.robot.set_joint_position(self.joint_name, target)
            self.robot.step(50)  # Allow time for head to move
            print(f"{self.name}: Moviendo cabeza a {target:.2f} rad")
    
    def filter_unique_objects(self, object_list):
        seen_names = set()
        unique_list = []
        for obj in object_list:
            position, name = obj
            if name not in seen_names:
                unique_list.append(obj)
                seen_names.add(name)
        return unique_list

    def update(self):
        if self.start_time is None:
            self.start_time = self.robot.get_time()

        # Check for timeout
        if (self.robot.get_time() - self.start_time) > self.timeout:
            print(f"{self.name}: Object recognition timed out after {self.timeout}s")
            return py_trees.common.Status.FAILURE

        # Check if we've completed the head sweep
        if self.current_idx >= len(self.waypoints):
            print(f"{self.name}: Barrido completado. Cabeza centrada.")
            return py_trees.common.Status.SUCCESS
            
        target = self.waypoints[self.current_idx]
        
        # Check if head has reached the target position
        current_pos = self.robot.get_joint_position(self.joint_name)

        if abs(current_pos - target) < self.tolerance:
            
            # Use robot's camera recognition to find objects in the current view
            objects = self.robot.camera.getRecognitionObjects()
            if objects:
                print(f"{self.name}: Camera sees {len(objects)} objects")

            # For each object detected, get its position in the world
            for obj in objects:
                try:
                    obj_id = obj.getId()
                    object_node = self.robot.supervisor.getFromId(obj_id) 
                    model_name = object_node.getField("name").getSFString()
                    
                    world_position = self.robot.planner.get_objects_positions([model_name])[0]
                    pos = world_position.get('position')
                    
                    # Filter objects with unknown or invalid positions
                    if pos is None or len(pos) != 3:
                        print(f"{self.name}: Ignorando '{model_name}' porque no tiene coordenadas válidas en Webots.")
                        continue
                    
                    # Store the detected object position and name in the list
                    self.object_positions.append((pos, model_name))
                    print(f"{self.name}: Found object: {model_name} at {pos}")
                    
                except Exception as e:
                    pass
            
            # Filter unique objects
            filtered_objects = self.filter_unique_objects(self.object_positions)
            
            # Publish the filtered list of object positions and names to the blackboard
            self.blackboard.set("filtered_object_positions", filtered_objects)
            
            self.current_idx += 1
            self.set_target()
            
        return py_trees.common.Status.RUNNING
    

class ObjectSelector(py_trees.behaviour.Behaviour):
    """
    Reads the list of detected objects from the blackboard and selects one for manipulation.
    """

    def __init__(self, name: str, robot: TiagoRobot):
        super(ObjectSelector, self).__init__(name)
        self.robot = robot
        
        self.blackboard = py_trees.blackboard.Blackboard()
        self.object_name = None
        self.target_position = None

    def initialise(self):
        
        # Reset variables at the start of the behavior
        self.object_name = None
        self.target_position = None
        print(f"{self.name}: Waiting for object list...")

    def update(self):

        # Read the list of detected objects from the blackboard
        filtered_objs = self.blackboard.get("filtered_object_positions")
        
        # Verify if the list exists and has at least one object
        if filtered_objs and len(filtered_objs) > 0:
            
            # Extract position and name of the first object in the list
            self.target_position = filtered_objs[0][0]
            self.object_name = filtered_objs[0][1]

            print(f"{self.name}: Object selected! -> {self.object_name}")
            
            # Publish selected object position and name to the blackboard for other behaviors
            self.blackboard.set("target_position", self.target_position)
            self.blackboard.set("object_name", self.object_name)
            
            # Remove selected object from the list
            filtered_objs.pop(0)
            
            # Update the blackboard with the remaining objects
            self.blackboard.set("filtered_object_positions", filtered_objs)
            
            return py_trees.common.Status.SUCCESS
        

        # If no objects are available, print a warning and fail
        else:
            print(f"{self.name}: No objects available for selection.")
            return py_trees.common.Status.FAILURE
        
    

class GraspController(py_trees.behaviour.Behaviour):
    """
    Grasp behaviour using magnet feedback
    
    State Machine:
    1. APPROACHING: Slowly close gripper until contact
    2. VERIFYING: Maintain grip with force validation
    """

    def __init__(self, name: str, robot_instance: TiagoRobot):
        super(GraspController, self).__init__(name)
        self.robot = robot_instance
        self.state = "APPROACHING"
        self.grip_width = 0.045  # Fully open
        self.verification_time = 0.5
        self.verification_start_time = None


    def initialise(self):
        self.state = "APPROACHING"
        self.grip_width = 0.045
        self.verification_start_time = None
        print(f"{self.name}: Starting grasp sequence")

        # Reset blackboard grasp_success
        try:
            blackboard = py_trees.blackboard.Blackboard()
            blackboard.set("grasp_success", False)
        except Exception as e:
            print(f"Error initializing blackboard: {e}")

    def update(self):

        # Read current gripper finger positions
        current_left = self.robot.get_joint_position('gripper_left_finger_joint')
        current_right = self.robot.get_joint_position('gripper_right_finger_joint')

        if self.state == "APPROACHING":
            # Move fingers closer slowly
            self.grip_width = max(0.0, self.grip_width - 0.0001) 
            self.robot.set_joint_position('gripper_left_finger_joint', self.grip_width)
            self.robot.set_joint_position('gripper_right_finger_joint', self.grip_width)

            # Check magnet presence and lock condition
            presence = self.robot.magnet.getPresence()
            if presence == 1 or presence == -1:
                self.robot.magnet.lock()
                
            if self.robot.magnet.isLocked():
                print(f"{self.name}: Contact detected. Magnet locked.")
                self.state = "VERIFYING"
                self.verification_start_time = self.robot.get_time()

        elif self.state == "VERIFYING":

            # Apply a bit more pressure
            target_width = max(0.0, self.grip_width - 0.1)
            self.robot.set_joint_position('gripper_left_finger_joint', target_width)
            self.robot.set_joint_position('gripper_right_finger_joint', target_width)

            current_time = self.robot.get_time()
            if current_time - self.verification_start_time >= self.verification_time:
                if self.robot.magnet.isLocked():
                    print(f"{self.name}: Grasp successful!")
                    try:
                        blackboard = py_trees.blackboard.Blackboard()
                        blackboard.set("grasp_success", True)
                    except Exception as e:
                        print(f"Error setting blackboard: {e}")
                    return py_trees.common.Status.SUCCESS
                else:
                    print(f"{self.name}: Grasp verification failed, retrying")
                    self.state = "APPROACHING"

        # Check for failure conditions
        if current_left < 0.005 and current_right < 0.005 and not self.robot.magnet.isLocked():
            print(f"{self.name}: Grasp failed - fingers closed but object not locked")
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
            self.robot.set_joint_position('gripper_left_finger_joint', 0.045)
            self.robot.set_joint_position('gripper_right_finger_joint', 0.045)


class OpenGripper(py_trees.behaviour.Behaviour):
    """
    Behavior to open the gripper using magnet control.
    """

    def __init__(self, name: str, robot_instance: TiagoRobot, open_position=0.045, timeout=2.0):
        super(OpenGripper, self).__init__(name)
        self.robot = robot_instance
        self.open_position = open_position
        self.timeout = timeout
        self.start_time = None
        self.gripper_opened = False

    def initialise(self):
        
        self.start_time = self.robot.get_time()
        self.gripper_opened = False
        print(f"{self.name}: Opening gripper to position {self.open_position}")

        # Set gripper to fully open position
        self.robot.set_joint_position('gripper_left_finger_joint', self.open_position)
        self.robot.set_joint_position('gripper_right_finger_joint', self.open_position)

    def update(self):
        
        current_time = self.robot.get_time()

        # Check if we've been running too long
        if current_time - self.start_time > self.timeout:
            print(f"{self.name}: Timeout reached, considering gripper opened")
            return py_trees.common.Status.SUCCESS

        # Check if gripper is already open based on magnet state and unlock
        if self.robot.magnet.isLocked():
            self.robot.magnet.unlock()
            print(f"{self.name}: Magnet unlocked, gripper opened")
            
        return py_trees.common.Status.SUCCESS


class LiftAndVerify(py_trees.behaviour.Behaviour):

    def __init__(self, name: str, robot_instance: TiagoRobot, timeout=2.0):
        super(LiftAndVerify, self).__init__(name)
        self.robot = robot_instance
        self.timeout = timeout

        self.start_time = None
        self.movement_started = False

    def initialise(self):
        self.start_time = self.robot.get_time()
        self.movement_started = False
        print(f"{self.name}: Starting lift sequence")

        # Keep the base stationary during the lift
        self.robot.set_base_velocity(0.0, 0.0)

    def update(self):

        current_time = self.robot.get_time()

        if not self.robot.magnet.isLocked():
            print(f"{self.name}: Object may have been dropped! Aborting lift.")
            
            # Update blackboard to indicate grasp failure
            blackboard = py_trees.blackboard.Blackboard()
            blackboard.set("grasp_success", False)
            return py_trees.common.Status.FAILURE

        # Start the lift movement if not started yet
        if not self.movement_started:
            
            # Offset lift from grasp position
            offset_lift = np.array([0.00, 0.0, 0.25]) # CHANGE THIS OFFSET TO ADJUST LIFT HEIGHT
            
            # Read current end-effector position
            current_grasp = self.robot.read_torso_and_arm_joints()
            prelift_position = self.robot.ik_chain.calculate_forward_kinematics(current_grasp)
            
            # Add a small step to ensure we start the lift from a stable position
            self.robot.step(200)
            
            path_lift = self.robot.planner.run_rrt_with_obstacles(
                x_init=tuple(prelift_position), 
                x_goal=tuple(prelift_position + offset_lift),
                headless = True
            )

            for waypoint in path_lift:
                
                # Convert to configuration space using IK
                self.target_angles = self.robot.ik_chain.calculate_inverse_kinematics(
                    waypoint, [0, 0, 1], orientation_mode="Z"
                )

                if not self.target_angles:
                    print(f"{self.name}: Failed to calculate IK solution.")
                    return py_trees.common.Status.FAILURE

                # Move arm joints to the new position using the robot instance
                for joint, angle in self.target_angles.items():
                    self.robot.set_joint_position(joint, angle)
                        
                self.robot.step(200)  # Allow some time for movement

            self.movement_started = True
            print(f"{self.name}: Arm moving to lift position")

        # Check for timeout - consider motion complete after timeout
        if current_time - self.start_time > self.timeout:
            print(f"{self.name}: Lift sequence completed")

            if self.robot.magnet.isLocked():
                print(f"{self.name}: Object securely held!")
                return py_trees.common.Status.SUCCESS
            else:
                print(f"{self.name}: Object lost during final lift position")
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING


class MoveArmJointsForwardKinematics(py_trees.behaviour.Behaviour):
    """
    Precise joint-space controller for arm positioning using the robot instance.
    """

    def __init__(self, name, robot_instance: TiagoRobot, joint_targets, tolerance=0.02, timeout=10.0):
        super(MoveArmJointsForwardKinematics, self).__init__(name)
        self.robot = robot_instance
        self.joint_targets = joint_targets
        self.tolerance = tolerance
        self.movement_complete = False
        self.print_interval = 20
        self.timeout = timeout
        self.start_time = None
        self.progress_time = None
        self.progress_threshold = 6.0  # Consider success if no progress for 6 seconds

    def initialise(self):
        self.movement_complete = False
        self.update_count = 0
        self.start_time = self.robot.get_time()
        self.progress_time = self.start_time
        self.last_errors = {}

        print(f"{self.name}: Moving arm to joint positions...")

        # Set all joint positions at initialization usando la instancia del robot
        for joint, target in self.joint_targets.items():
            if joint in self.robot.motors:
                self.robot.set_joint_position(joint, target)
                # Obtenemos el valor del sensor a través del robot
                current_val = self.robot.get_joint_position(joint)
                self.last_errors[joint] = abs(target - current_val)
                print(f"Setting {joint} to position {target}")

    def update(self):
        if self.movement_complete:
            return py_trees.common.Status.SUCCESS

        current_time = self.robot.get_time()

        # Check for timeout
        if current_time - self.start_time > self.timeout:
            print(f"{self.name} timed out after {self.timeout} seconds, considering it complete")
            self.movement_complete = True
            return py_trees.common.Status.SUCCESS
        
        # Check if we're making progress
        making_progress = False
        all_joints_in_position = True

        for joint, target in self.joint_targets.items():
            # Verificamos si el sensor existe en la instancia del robot
            if joint not in self.robot.sensors:
                continue
            
            # Calculate error between target and current position 
            current_position = self.robot.get_joint_position(joint)
            error = abs(target - current_position)

            # Check if this joint is still moving
            if joint in self.last_errors:
                if abs(self.last_errors[joint] - error) > 0.00005:
                    making_progress = True
                self.last_errors[joint] = error

            # If the joint is outside the tolerance, keep it moving
            if error > self.tolerance:
                self.robot.set_joint_position(joint, target)
                all_joints_in_position = False

                # Print detailed information for joints that are far from target
                if error > 0.1 and self.update_count % self.print_interval == 0:
                    print(f"Joint {joint} at {current_position:.4f}, target: {target:.4f}, error: {error:.4f}")

        self.robot.step(400)

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

            self.movement_complete = True
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING


class CheckHardwareStatus(py_trees.behaviour.Behaviour):
    """
    Robot hardware integrity verification behavior.
    
    Systematically checks all required hardware components using the robot instance:
    - Motor controllers
    - Position sensors
    - Camera subsystem
    - Navigation sensors (GPS, compass)
    """

    def __init__(self, name, robot_instance: TiagoRobot):
        super(CheckHardwareStatus, self).__init__(name)
        self.robot = robot_instance
        self.required_devices = {
            'motors': ['torso_lift_joint', 'arm_1_joint', 'gripper_left_finger_joint'],
            'sensors': ['torso_lift_joint', 'arm_1_joint', 'gripper_left_finger_joint']
        }

    def update(self):
        # 1. Check motors en el diccionario del robot
        for motor_name in self.required_devices['motors']:
            if motor_name not in self.robot.motors:
                print(f"ERROR: Required motor '{motor_name}' not found in robot.motors")
                return py_trees.common.Status.FAILURE

        # 2. Check sensors en el diccionario del robot
        for sensor_name in self.required_devices['sensors']:
            if sensor_name not in self.robot.sensors:
                print(f"ERROR: Required sensor '{sensor_name}' not found in robot.sensors")
                return py_trees.common.Status.FAILURE

        # 3. Check other devices
        if not self.robot.camera:
            print("ERROR: Camera not found on robot instance")
            return py_trees.common.Status.FAILURE

        if not self.robot.gps:
            print("ERROR: GPS not found on robot instance")
            return py_trees.common.Status.FAILURE

        if not self.robot.compass:
            print("ERROR: Compass not found on robot instance")
            return py_trees.common.Status.FAILURE

        print("All hardware components verified successfully on TiagoRobot instance")
        return py_trees.common.Status.SUCCESS


class RuntimeMonitor:
    """
    System health and performance monitoring subsystem.
    """

    def __init__(self, robot_instance: TiagoRobot, log_interval=10.0):
        self.robot = robot_instance
        self.start_time = self.robot.get_time()
        self.last_log_time = self.start_time
        self.log_interval = log_interval
        self.joint_position_history = {}
        self.motor_velocity_history = {}

        # Add GPS position tracking
        self.position_history = []
        self.max_position_history = 100
        self.last_position = None
        
        # Access GPS position at initialization using the robot instance
        if self.robot.gps:
            self.last_position = self.robot.gps.getValues()

    def update(self):
        current_time = self.robot.get_time()

        # Track joint positions
        for joint_name in self.robot.sensors:
            if joint_name not in self.joint_position_history:
                self.joint_position_history[joint_name] = []
            
            # Obtain joint position using the robot instance
            val = self.robot.get_joint_position(joint_name)
            self.joint_position_history[joint_name].append(val)
            
            # Limit history length
            if len(self.joint_position_history[joint_name]) > 100:
                self.joint_position_history[joint_name].pop(0)

        # Track GPS position
        if self.robot.gps:
            current_pos = self.robot.gps.getValues()
            self.position_history.append(current_pos)
            if len(self.position_history) > self.max_position_history:
                self.position_history.pop(0)

            # Calculate movement speed
            if self.last_position:
                dx = current_pos[0] - self.last_position[0]
                dy = current_pos[1] - self.last_position[1]
                distance = np.sqrt(dx * dx + dy * dy)
                
                dt = self.robot.timestep / 1000.0
                speed = distance / dt if dt > 0 else 0
                self.last_position = current_pos

        # Log periodically
        if current_time - self.last_log_time >= self.log_interval:
            self.log_robot_status()
            self.last_log_time = current_time

    def log_robot_status(self):
        """
        Log important robot metrics with enhanced GPS and compass data
        """
        
        # Robot position and orientation
        if self.robot.gps and self.robot.compass:
            pos = self.robot.gps.getValues()
            compass_vals = self.robot.compass.getValues()
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

        # Battery status
        battery_level = self.robot.supervisor.getBatteryValue() if hasattr(self.robot.supervisor, "getBatteryValue") else "N/A"
        print(f"Battery level: {battery_level}")

        # Check for any excessive joint velocities or oscillations
        for joint_name, history in self.joint_position_history.items():
            if len(history) > 20:
                # Check for oscillations by looking at position changes
                changes = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
                if sum(changes) > 0.5 and max(changes) < 0.1:
                    print(f"WARNING: Possible oscillation in joint {joint_name}")

class NavigationWithRRT(py_trees.behaviour.Behaviour):
    """
    Behavior to navigate the TIAGo base to a target (x, y, heading).
    Uses AStar for planning and Pure Pursuit for path following.
    """
    def __init__(self, name, robot, target_dict, map_metadata_path="occupancy_map.json"):
        super(NavigationWithRRT, self).__init__(name)
        # self.robot is of the TiagoRobot custom class
        # self.webots_robot is of the Robot class (from Webots), and we use it to access devices
        self.robot = robot
        self.webots_robot = self.robot.supervisor
        self.target_dict = target_dict
        self.map_path = map_metadata_path
        self.phase = "PLANNING" # Phases: PLANNING, FOLLOWING, ROTATING
        self.blackboard = py_trees.blackboard.Blackboard()

    def initialise(self):
        print("Initializing robot and devices...")
        # self.robot is of the TiagoRobot custom class
        # webots_robot is of the Robot class (from Webots), and we use it to access devices
        
        timestep = int(self.robot.timestep)
        dt_sec = timestep / 1000.0  # dt in seconds for the PI controller

        # Robot wheels (actuators)
        left_motor = self.webots_robot.getDevice("wheel_left_joint")
        right_motor = self.webots_robot.getDevice("wheel_right_joint")
        left_motor.setPosition(float('inf'))
        right_motor.setPosition(float('inf'))
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        # Robot localization devices (sensors)
        gps = self.webots_robot.getDevice("gps")
        gps.enable(timestep)
        compass = self.webots_robot.getDevice("compass")
        compass.enable(timestep)

        # Initial arm configuration
        initial_pose = {
            'torso_lift_joint': 0.3, 'arm_1_joint': 0.71, 'arm_2_joint': 1.02,
            'arm_3_joint': -2.815, 'arm_4_joint': 1.011, 'arm_5_joint': 0,
            'arm_6_joint': 0, 'arm_7_joint': 0,
            'gripper_left_finger_joint': 0.045, 'gripper_right_finger_joint': 0.045,
            'head_1_joint': 0, 'head_2_joint': 0
        }
        
        print("Moving the robot arm to the initial pose...")
        for joint_name, position in initial_pose.items():
            try:
                self.robot.set_joint_position(joint_name, position)
            except Exception as e:
                print(f"Warning: Could not initialize motor/sensor '{joint_name}': {e}")

        while self.robot.step(timestep) != -1:
            all_joints_in_position = True
            for joint_name, target_pos in initial_pose.items():
                current_pos = self.robot.get_joint_position(joint_name)
                if abs(current_pos - target_pos) > ARM_POSITION_TOLERANCE:
                    all_joints_in_position = False
                    break
            if all_joints_in_position:
                print("Arm is in position")
                break

        print(f"{self.name}: Initializing navigation to {self.target_dict}...")
        self.phase = "PLANNING"
        
        # Load Map and Metadata
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
        
        #using logic from path planning, inflate radius reduced 0.5 -> 0.45
        print("Inflating grid map...")
        map_ = Grid(bounds=[[0, width], [0, height]])

        map_.type_map[:, :] = TYPES.FREE
        map_.type_map[img < 128] = TYPES.OBSTACLE

        inflate_radius_m = 0.45
        inflate_radius_px = int(np.ceil(inflate_radius_m / MAP_RESOLUTION))
        map_.inflate_obstacles(radius=inflate_radius_px)
        print(f"Obstacles inflated by {inflate_radius_px} pixels.")

        # Plan Path
        print("Beginning path planning")
        current_target = self.target_dict
        GOAL_WORLD = np.array([current_target["x"], current_target["y"]])
        TARGET_HEADING = current_target["heading"]

        gps_vec = self.webots_robot.getDevice("gps").getValues()
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
            self.phase = "FAILED"

        else:
            path_world = convert_path_to_world(path, MAP_ORIGIN, MAP_RESOLUTION)
            self.trajectory = Trajectory(path_world[:, 0], path_world[:, 1], look_ahead_dist=L)
            self.PI_angular = PI(dt=self.robot.timestep/1000.0, kp=PI_KP, ki=PI_KI)
            self.phase = "FOLLOWING"
            print(f"Path found! {len(path)} points. FOLLOWING")

        """
        if path:
            self.trajectory = Trajectory(path_world[:, 0], path_world[:, 1], look_ahead_dist=L)
            self.PI_angular = PI(dt=self.robot.timestep/1000.0, kp=PI_KP, ki=PI_KI)
            self.phase = "FOLLOWING"
            print(f"{self.name}: Path planned. Following...")
        else:
            self.phase = "FAILED" 
        """

    def update(self):
        if self.phase == "FAILED":
            return py_trees.common.Status.FAILURE
        
        # Get Current State
        gps_vals = self.webots_robot.getDevice("gps").getValues()
        compass_vals = self.webots_robot.getDevice("compass").getValues()
        curr_pos = [gps_vals[0], gps_vals[1]]
        yaw = math.atan2(compass_vals[0], compass_vals[1])
        goal_pos = [self.target_dict["x"], self.target_dict["y"]]

        if self.phase == "FOLLOWING":
            if getDistance(curr_pos, goal_pos) < GOAL_TOLERANCE:
                self.phase = "ROTATING"
                return py_trees.common.Status.RUNNING
            
            target_pt = self.trajectory.getTargetPoint(curr_pos)
            target_yaw = math.atan2(target_pt[1] - curr_pos[1], target_pt[0] - curr_pos[0])
            self._apply_control(normalize_angle(target_yaw - yaw), TARGET_VELOCITY)
            return py_trees.common.Status.RUNNING

        if self.phase == "ROTATING":
            yaw_error = normalize_angle(self.target_dict["heading"] - yaw)
            if abs(yaw_error) < 0.05: # HEADING_TOLERANCE
                self._stop_robot()
                return py_trees.common.Status.SUCCESS
            
            self._apply_control(yaw_error, 0.0)
            return py_trees.common.Status.RUNNING

        return py_trees.common.Status.FAILURE

    def _apply_control(self, yaw_error, v_target):
        w_target = self.PI_angular.control(yaw_error)
        v_l, v_r = convert_diff_drive(v_target, w_target, WHEEL_BASE, WHEEL_RADIUS)
        # Limit velocities
        max_v = max(abs(v_l), abs(v_r))
        if max_v > MAX_WHEEL_VELOCITY:
            scale = MAX_WHEEL_VELOCITY / max_v
            v_l, v_r = v_l * scale, v_r * scale
        self.robot.set_base_velocity(v_l,v_r)
        
    def _stop_robot(self):
        self.robot.set_base_velocity(0.0,0.0)

    