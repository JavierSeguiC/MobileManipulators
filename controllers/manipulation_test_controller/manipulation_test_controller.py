#!/usr/bin/env python3
import py_trees
from robot import TiagoRobot
from behaviours import (
    MoveArmTrajectoryRRT,
    EnhancedObjectRecognizer,
    ObjectSelector,
    GraspController,
    LiftAndVerify,
    OpenGripper,
    MoveArmJointsForwardKinematics,
    CheckHardwareStatus,
    RuntimeMonitor
)

# ==========================================
# TEST CONFIGURATION
# ==========================================
# Change this variable to test different parts of the manipulation pipeline.
# Options: "PICK", "PLACE", or "BOTH"
TEST_MODE = "PICK"  

# ==========================================
# PREDEFINED POSITIONS
# ==========================================
starting_position = {
    'torso_lift_joint': 0.1, 
    'arm_1_joint': 0.07, 
    'arm_2_joint': 0.7,
    'arm_3_joint': -1.45, 
    'arm_4_joint': 1.4, 
    'arm_5_joint': 0.6,
    'arm_6_joint': 0, 
    'arm_7_joint': 0, 
    'gripper_left_finger_joint': 0.045,
    'gripper_right_finger_joint': 0.045, 
    'head_1_joint': 0, 
    'head_2_joint': -0.5
}

# The position the robot uses to look for the object
planning_position = {
    "torso_lift_joint": 0.35,  # Max height to get a good view of the table
    "arm_1_joint": 0.07,
    "arm_2_joint": 0.75,
    "arm_3_joint": -1.75,
    "arm_4_joint": 0.0,
    "arm_5_joint": -1.5,
    "arm_6_joint": 0.0,
    "arm_7_joint": 0.0,
    "head_2_joint": 0       # Tilted down to look at the table
}

manipulation_position = {
    "torso_lift_joint": 0.35,  # Suitable height for manipulation
    'arm_1_joint': 0.07, 
    'arm_2_joint': 0.7,
    'arm_3_joint': -1.45, 
    'arm_4_joint': 1.4, 
    'arm_5_joint': 0.6,
    'arm_6_joint': 0, 
    'arm_7_joint': 0, 
    'gripper_left_finger_joint': 0.045,
    'gripper_right_finger_joint': 0.045, 
    'head_1_joint': 0, 
    "head_2_joint": -0.5       # Look down   
}

def create_behavior_tree(robot):
    """Creates a simplified sequence just for testing manipulation."""
    root = py_trees.composites.Sequence(f"Test Sequence: {TEST_MODE}", memory=True)

    """
    # 1. Initialization (Always runs)
    init_seq = py_trees.composites.Sequence("Initialization", memory=True)
    init_seq.add_children([
        CheckHardwareStatus("Check Hardware", robot),
        MoveArmJointsForwardKinematics("Initial Safe Position", robot, starting_position)
    ])
    root.add_child(init_seq)
    """

    # 2. PICK SEQUENCE
    if TEST_MODE in ["PICK", "BOTH"]:
        pick_test_seq = py_trees.composites.Sequence("Test Pick", memory=True)
        
        # Perception
        pick_test_seq.add_children([
            EnhancedObjectRecognizer("Recognize Objects", robot, timeout=60.0),
            ObjectSelector("Select Object", robot)
        ])
        
        # Execution
        pick_test_seq.add_children([
            MoveArmJointsForwardKinematics("Adjust Height for Manipulation", robot, manipulation_position),
            MoveArmTrajectoryRRT(
                name="Approach Object",
                robot=robot,
                use_target_from_blackboard=True,
                offsets=[[0.0, 0.0, 0.40], [0.0, 0.0, 0.09]]
            ),
            GraspController("Grasp Object", robot),
            LiftAndVerify("Lift Object", robot)
        ])
        root.add_child(pick_test_seq)

    # 3. PLACE SEQUENCE
    if TEST_MODE in ["PLACE", "BOTH"]:
        place_test_seq = py_trees.composites.Sequence("Test Place", memory=True)
        
        place_test_seq.add_children([
            MoveArmTrajectoryRRT(
                name="Move to Drop Position",
                robot=robot,
                use_target_from_blackboard=False,
                fixed_target=[0.79, 0.0, 1.15], # Drop coordinates relative to robot base
                offsets=[[0.0, 0.0, 0.0]]
            ),
            OpenGripper("Drop Object", robot),
            MoveArmJointsForwardKinematics("Return Arm to Safe Pos", robot, starting_position)
        ])
        root.add_child(place_test_seq)

    bt = py_trees.trees.BehaviourTree(root)
    # bt.visitors.append(py_trees.visitors.DebugVisitor()) # Uncomment for extremely detailed tree logs
    return bt

def main():
    print(f"Initializing Manipulation Tester in {TEST_MODE} mode...")
    robot = TiagoRobot("tiago_kitchen_finalproject.wbt") 
    
    #monitor = RuntimeMonitor(robot_instance=robot, log_interval=10.0)
    
    bt = create_behavior_tree(robot)
    bt.setup(timeout=15)
    bt.tick()

    print("Starting execution...")
    while robot.supervisor.step(robot.timestep) != -1:
        #monitor.update()
        bt.tick()
        
        if bt.root.status == py_trees.common.Status.SUCCESS:
            print(f"\n>>> TEST {TEST_MODE} COMPLETED SUCCESSFULLY! <<<")
            break
        elif bt.root.status == py_trees.common.Status.FAILURE:
            print(f"\n>>> TEST {TEST_MODE} FAILED. <<<")
            break

if __name__ == '__main__':
    main()