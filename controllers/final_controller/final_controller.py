#!/usr/bin/env python3
import py_trees
import math
from robot import TiagoRobot
from behaviours import (
    NavigationWithRRT, 
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

# Constants/Targets
TARGETS = {
    "green_basket": {"x": -1.12, "y": 0.22, "heading": math.radians(89.02)},
    "nocilla": {"x": 0.64, "y": -0.15, "heading": math.radians(0)},
    "nutella": {"x": 0.28, "y": -1.53, "heading": math.radians(-137.53)},
    "home": {"x": -0.93, "y": -3.14, "heading": math.radians(1.57)}
}

starting_position = {
    'torso_lift_joint': 0.35, 
    'arm_1_joint': 0.71, 
    'arm_2_joint': 1.02,
    'arm_3_joint': -2.815, 
    'arm_4_joint': 1.011, 
    'arm_5_joint': 0,
    'arm_6_joint': 0, 
    'arm_7_joint': 0,
    'gripper_left_finger_joint': 0.045, 
    'gripper_right_finger_joint': 0.045,
    'head_1_joint': 0, 
    'head_2_joint': 0
}
"""
starting_position = {
    'torso_lift_joint': 0.35, 
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
"""
"""
planning_position = {
    "torso_lift_joint": 0.35,
    "arm_1_joint": 1.20,
    "arm_2_joint": 1.02,
    "arm_3_joint": 0.29,
    "arm_4_joint": 1.12,
    "arm_5_joint": 1.71,
    "arm_6_joint": -1.39,
    "arm_7_joint": 1.53
}"""

manipulation_position = {
    "torso_lift_joint": 0.35,  
    "arm_1_joint": 0.94,       
    "arm_2_joint": 1.02,       
    "arm_3_joint": -1.06,
    "arm_4_joint": 1.43,       
    "arm_5_joint": 1.09,     
    "arm_6_joint": -1.39,
    "arm_7_joint": 1.80,
    "head_2_joint": -0.6       
}

carrying_position = {
    "torso_lift_joint": 0.35,  
    "arm_1_joint": 0.94,       
    "arm_2_joint": 1.02,       
    "arm_3_joint": -1.3,
    "arm_4_joint": 1.50,       
    "arm_5_joint": 1.09,     
    "arm_6_joint": -1.39,
    "arm_7_joint": 1.80,
    "head_2_joint": 0         
}


def create_behavior_tree(robot):
    """Creates the sequence for the pick and place task."""
    root = py_trees.composites.Sequence("Pick and Place Sequence", memory=True)

    # 1. Initialization
    init_seq = py_trees.composites.Sequence("Initialization", memory=True)
    init_seq.add_children([
        CheckHardwareStatus("Check Hardware", robot),
        MoveArmJointsForwardKinematics("Initial Safe Position", robot, starting_position)
    ])
    root.add_child(init_seq)

    objects_to_pick = ["nocilla", "nutella"]    #list of objects to pick and place

    for obj_name in objects_to_pick:    #iterate thru the list of objects to pick and place
        obj_seq = py_trees.composites.Sequence(f"Process {obj_name.capitalize()}", memory=True)

        # 2. Navigate to the object location
        go_to_object = NavigationWithRRT(f"Navigate to {obj_name.capitalize()}", robot, TARGETS[obj_name])
        
        # 3. Perception and Planning (Look at the table and find the exact coordinates)
        prep_seq = py_trees.composites.Sequence("Perception & Planning", memory=True)
        prep_seq.add_children([
            #MoveArmJointsForwardKinematics("Move to Planning Position", robot, planning_position),
            EnhancedObjectRecognizer("Recognize Objects", robot, timeout=60.0),
            ObjectSelector("Select Object", robot)
        ])

        # 4. Pick Sequence (Approach, Grasp, Lift)
        pick_seq = py_trees.composites.Sequence("Pick Sequence", memory=True)
        pick_seq.add_children([
            MoveArmJointsForwardKinematics("Pose for Manipulation", robot, manipulation_position),
            MoveArmTrajectoryRRT(
                name="Approach Object",
                robot=robot,
                use_target_from_blackboard=True,
                offsets=[[0.0, 0.0, 0.40], [0.0, 0.0, 0.09]]
            ),
            GraspController("Grasp Object", robot),
            LiftAndVerify("Lift Object", robot),
            MoveArmJointsForwardKinematics("Move to Carry Position", robot, carrying_position)
        ])

        # 5. Navigate to Basket
        go_to_basket = NavigationWithRRT("Navigate to Basket", robot, TARGETS["green_basket"])

        # 6. Place Sequence (Drop & Reset)
        place_seq = py_trees.composites.Sequence("Place Sequence", memory=True)
        place_seq.add_children([
            MoveArmJointsForwardKinematics("Pose for Drop Manipulation", robot, manipulation_position),
            
            OpenGripper("Drop Object", robot),
            MoveArmJointsForwardKinematics("Return Arm to Safe Pos", robot, starting_position)
        ])
        """ 
        Removed from place sequence
            MoveArmTrajectoryRRT(
                name="Move to Drop Position",
                robot=robot,
                use_target_from_blackboard=False,
                fixed_target=[0.79, 0.0, 1.15], # Drop coordinates over basket
                offsets=[[0.0, 0.0, 0.0]]
            ),
        """
        # Assemble the full sequence for this object
        obj_seq.add_children([go_to_object, prep_seq, pick_seq, go_to_basket, place_seq])
        root.add_child(obj_seq)

    # 7. Return Home
    go_home = NavigationWithRRT("Return Home", robot, TARGETS["home"])
    root.add_child(go_home)

    bt = py_trees.trees.BehaviourTree(root)
    bt.visitors.append(py_trees.visitors.DebugVisitor())
    return bt

def main():
    print("Initializing TIAGo Final Controller...")
    # Initialize robot with the project world file
    robot = TiagoRobot("tiago_kitchen_finalproject.wbt")

    # Initialize system health monitor
    monitor = RuntimeMonitor(robot_instance=robot, log_interval=10.0) 
    
    # Create and setup the tree
    print("Creating behaviour tree")
    bt = create_behavior_tree(robot)
    bt.setup(timeout=15)

    bt.tick()

    last_print_time = robot.get_time()
    print_interval = 5.0

    print("Starting Behavior Tree execution...")
    while robot.supervisor.step(robot.timestep) != -1:
        monitor.update()
        bt.tick()
        
        current_time = robot.get_time()

        # Optional: Print tree status every few seconds for debugging
        if current_time - last_print_time >= print_interval:
            print("\n" + "="*40)
            print(f"Behavior Tree Status at {current_time:.1f}s:")
            print(py_trees.display.ascii_tree(bt.root))
            print("="*40)
            last_print_time = current_time

        if bt.root.status == py_trees.common.Status.SUCCESS:
            print("\n>>> Task Completed Successfully! <<<")
            break
        elif bt.root.status == py_trees.common.Status.FAILURE:
            print("\n>>> Task Failed. <<<")
            break

if __name__ == '__main__':
    main()