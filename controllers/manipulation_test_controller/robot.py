from controller import Supervisor
from inverse_kinematics import InverseKinematics
from arm_planner import *

class TiagoRobot:
    """
    Main class that encapsulates the Webots API for the TIAGo robot.
    Handles initialization of all sensors, motors, kinematics and planners.
    """
    def __init__(self, wbt_world_path):
        print("Initializing TiagoRobot and devices...")
        self.supervisor = Supervisor()
        self.wbt_world_path = wbt_world_path
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # Dictionaries for general joints
        self.motors = {}
        self.sensors = {}
        

        # Initialize all subsystems
        self._setup_arm_and_torso()
        self._setup_wheels()
        self._setup_sensors()
        self._setup_ik_and_planner()


    def _setup_arm_and_torso(self):
        """Initializes motors and position sensors for joints."""
        part_names = [
            "head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
            "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint",
            "arm_6_joint", "arm_7_joint", "gripper_left_finger_joint", "gripper_right_finger_joint"
        ]

        for joint_name in part_names:
            try:
                # Setup motor
                motor = self.supervisor.getDevice(joint_name)
                motor.setVelocity(0.3)
                self.motors[joint_name] = motor
                
                # Setup sensor (with exception for gripper fingers)
                if joint_name == "gripper_left_finger_joint":
                    sensor_name = "gripper_left_sensor_finger_joint"
                elif joint_name == "gripper_right_finger_joint":
                    sensor_name = "gripper_right_sensor_finger_joint"
                else:
                    sensor_name = f"{joint_name}_sensor"
                    
                sensor = self.supervisor.getDevice(sensor_name)
                sensor.enable(self.timestep)
                self.sensors[joint_name] = sensor
                
            except Exception as e:
                print(f"Warning: Could not initialize motor/sensor '{joint_name}': {e}")

        # Enable force feedback for gripper
        self.motors['gripper_left_finger_joint'].enableForceFeedback(self.timestep)
        self.motors['gripper_right_finger_joint'].enableForceFeedback(self.timestep)

    def _setup_wheels(self):
        """Initializes wheel motors for base navigation."""
        self.left_motor = self.supervisor.getDevice('wheel_left_joint')
        self.right_motor = self.supervisor.getDevice('wheel_right_joint')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def _setup_sensors(self):
        """Initializes RGB camera, Lidar, GPS, Compass and Magneto."""
        # RGB camera and recognition
        self.camera = self.supervisor.getDevice('Astra rgb')
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)

        # Depth camera
        self.depth = self.supervisor.getDevice('Astra depth')
        self.depth.enable(self.timestep)

        # Navigation
        self.gps = self.supervisor.getDevice('gps')
        self.gps.enable(self.timestep)

        self.compass = self.supervisor.getDevice('compass')
        self.compass.enable(self.timestep)

        # Lidar
        self.lidar = self.supervisor.getDevice("Hokuyo URG-04LX-UG01")
        self.lidar.enable(self.timestep)

        # Magneto (Gripper)
        self.magnet = self.supervisor.getDevice("active_connector")
        self.magnet.enablePresence(self.timestep)

    def _setup_ik_and_planner(self):
        """Initializes inverse kinematics and RRT planner."""
        # IK Chain
        self.ik_chain = InverseKinematics("Robot.urdf", self.sensors)

        # RRT Planner
        file_path = Path(__file__).resolve()
        wbt_paths = [file_path.parent.parent.parent / "worlds" / self.wbt_world_path]
        
        self.planner = Planner(
            planner_limits = np.array([(-10.0, 10.0), (-10.0, 10.0), (0.0, 10)]),
            wbt_paths = wbt_paths,
            rrt_q = 0.02,
            rrt_r = 0.01,
            rrt_max_samples = 2000,
            rrt_prc = 0.1,
            headless = True
        )

    # ---------------------------------------------------------
    # UTILITY METHODS (To be used by Behaviors)
    # ---------------------------------------------------------
    def step(self, num_steps=1):
        """
        Advances the simulation. 
        If passed a number, it advances that many steps automatically.
        """
        for _ in range(num_steps):
            # If Webots closes or stops, returns -1
            if self.supervisor.step(self.timestep) == -1:
                return -1
        return 0

    def get_time(self):
        """Returns the current simulation time."""
        return self.supervisor.getTime()

    def set_joint_position(self, joint_name, position):
        """Moves a motor to a specific position."""
        if joint_name in self.motors:
            self.motors[joint_name].setPosition(position)

    def get_joint_position(self, joint_name):
        """Reads the current value of a joint sensor."""
        if joint_name in self.sensors:
            return self.sensors[joint_name].getValue()
        return 0.0
    
    def read_torso_and_arm_joints(self):
        """
        Reads and returns the current positions of torso and arm joints.
        Essential for calculating Forward Kinematics.
        """
        joints_of_interest = [
            "torso_lift_joint", 
            "arm_1_joint", "arm_2_joint", "arm_3_joint", 
            "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"
        ]
        
        # Extract values using the existing method
        # Returns a dictionary {joint_name: current_angle}
        current_positions = {}
        for joint in joints_of_interest:
            current_positions[joint] = round(self.get_joint_position(joint), 3)
            
        return current_positions

    def set_base_velocity(self, left_vel, right_vel):
        """Moves the robot's mobile base."""
        self.left_motor.setVelocity(left_vel)
        self.right_motor.setVelocity(right_vel)