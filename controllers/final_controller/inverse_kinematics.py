from ikpy.chain import Chain
import urdf_parser_py.urdf as urdf_model
import numpy as np

class InverseKinematics:
    def __init__(self, urdf_path, sensors):

        self.urdf_path = urdf_path
        self.urdf_root = urdf_model.URDF.from_xml_file(self.urdf_path)

        self.base_elements = [
            "base_link",
            "base_link_Torso_joint",
            "Torso",
            "torso_lift_joint",
            "torso_lift_link",
            "torso_lift_link_TIAGo front arm_joint",
            "TIAGo front arm_3",
            "arm_1_joint",
            "TIAGo front arm_3", "arm_2_joint",
            "arm_2_link",
            "arm_3_joint",
            "arm_3_link",
            "arm_4_joint",
            "arm_4_link",
            "arm_5_joint",
            "arm_5_link",
            "arm_6_joint",
            "arm_6_link",
            "arm_7_joint",
            "arm_7_link",
            "arm_7_link_wrist_ft_tool_link_joint",
            "wrist_ft_tool_link",
            "wrist_ft_tool_link_front_joint"
        ]

        self.part_names = [
            "head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
            "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint",
            "arm_6_joint", "arm_7_joint", "wheel_left_joint", "wheel_right_joint",
            "gripper_left_finger_joint", "gripper_right_finger_joint"
        ]   

        # Parse joint limits
        self.joint_limits = {
            joint.name: {
                "lower": joint.limit.lower,
                "upper": joint.limit.upper,
                "velocity": joint.limit.velocity
            }
            for joint in self.urdf_root.joint_map.values()
            if joint.limit is not None
        }

        # Create the IK chain
        self.chain = self.create_ik_chain()

        self.sensors = sensors

    # Create IK chain
    def create_ik_chain(self):
        """Create an improved IK chain with better configuration"""
        # First create the chain without an active links mask

        chain = Chain.from_urdf_file(
            self.urdf_path,
            base_elements=self.base_elements,
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
        #print("IK chain created successfully",chain.links)
        return Chain(
            links=chain.links,
            active_links_mask=active_links_mask,
            name="tiago_arm"
        )

    def clamp_joint_angles(self, joint_angles):
        """Clamp joint angles to their URDF limits based on the IK chain links"""
        # Start with a copy of the original angles
        clamped_angles = list(joint_angles)
        
        # Iterate through the chain links to match names with limits
        for i, link in enumerate(self.chain.links):
            # Make sure we don't go out of bounds of the provided joint_angles list
            if i < len(clamped_angles):
                if link.name in self.joint_limits:
                    lower = self.joint_limits[link.name]['lower']
                    upper = self.joint_limits[link.name]['upper']
                    
                    # Clamp the value using numpy
                    clamped_angles[i] = float(np.clip(clamped_angles[i], lower, upper))
                    
        return clamped_angles
    
    def calculate_inverse_kinematics(self, target_position, target_rotation, orientation_mode="Z", offset_x=0.0, offset_y=0.0):
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
            self.sensors[joint.name].getValue() if joint.name in self.sensors else 0.0
            for joint in self.chain.links
        ]

        initial_position = self.clamp_joint_angles(initial_position)  # Clamp to joint limits
        print(f"Initial joint positions (clamped): {initial_position}")

        # Perform the IK calculation
        try:
            ik_results = self.chain.inverse_kinematics(
                target_position=final_target,
                initial_position=initial_position,
                target_orientation=target_rotation,
                orientation_mode=orientation_mode
            )
            print("IK solution found successfully")
            return {
                link.name: ik_results[i]
                for i, link in enumerate(self.chain.links)
                if link.name in self.part_names
            }
        except ValueError as e:
            print(f"IK solver error: {e}")
            return None

    def calculate_forward_kinematics(self, joint_angles_dict):
        """
        Computes the end effector position (x, y, z) given a set of joint angles.
        
        Args:
            joint_angles_dict (dict): Mapping of joint names to angles (radians). 
                                    Example: {'torso_lift_joint': 0.1, ...}
            
        Returns:
            list: [x, y, z] position of the end effector in world coordinates.
        """
        
        # 1. Prepare the complete joint vector for IKPy
        # IKPy needs a list with a value for EACH link in the chain (including fixed/base)
        full_joints_vector = []
        
        for link in self.chain.links:
            if link.name in joint_angles_dict:
                # If we pass a value for this joint, use it
                full_joints_vector.append(joint_angles_dict[link.name])
            elif link.name in self.sensors:
                # OPTIONAL: If we don't pass a value, use the current sensor value (current robot state)
                # If you prefer it to be 0 by default, change this to: full_joints_vector.append(0.0)
                full_joints_vector.append(self.sensors[link.name].getValue())
            else:
                # For fixed links or unknown ones, set to 0.0
                full_joints_vector.append(0.0)
                
        # print(f"FK Input Vector: {full_joints_vector}") # Uncomment for debugging

        # 2. Calculate the homogeneous transformation matrix (4x4)
        transformation_matrix = self.chain.forward_kinematics(full_joints_vector)
        
        # 3. Extract the position (X, Y, Z)
        # The matrix is:
        # [ R11 R12 R13  X ]
        # [ R21 R22 R23  Y ]
        # [ R31 R32 R33  Z ]
        # [  0   0   0   1 ]
        position = transformation_matrix[:3, 3]
        
        return list(position)
