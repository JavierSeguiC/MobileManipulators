#!/usr/bin/env python3
import py_trees
import math
from robot import TiagoRobot
from behaviours import NavigationWithRRT, MoveArmTrajectoryRRT # Assuming MoveArm is in behaviours.py

# Constants/Targets
TARGETS = {
    "green_basket": {"x": -1.24, "y": 0.48, "heading": math.radians(89.02)},
    "nocilla": {"x": 0.69, "y": -0.31, "heading": math.radians(-3.87)},
    "nutella": {"x": 0.28, "y": -1.63, "heading": math.radians(-137.53)},
    "home": {"x": -0.93, "y": -3.14, "heading": math.radians(1.57)}
}


def create_behavior_tree(robot):
    """Creates the sequence for the pick and place task."""
    root = py_trees.composites.Sequence("Pick and Place Sequence", memory=True)

    # 1. Navigate to the object (Nocilla)
    go_to_object = NavigationWithRRT("Navigate to Nocilla", robot, TARGETS["nocilla"])
    
    # 2. Pick the object (Placeholder for manipulation logic)
    # pick_object = PickObjectBehaviour("Pick Nocilla", robot) 

    # 3. Navigate to the delivery basket
    go_to_basket = NavigationWithRRT("Navigate to Basket", robot, TARGETS["green_basket"])

    # 4. Place the object (Placeholder)
    # place_object = PlaceObjectBehaviour("Place Nocilla", robot)

    # 5. Return Home
    go_home = NavigationWithRRT("Return Home", robot, TARGETS["home"])

    root.add_children([go_to_object, go_to_basket, go_home])
    return py_trees.trees.BehaviourTree(root)

def main():
    print("Initializing TIAGo Final Controller...")
    # Initialize robot with the project world file
    robot = TiagoRobot("tiago_kitchen_finalproject.wbt") 
    
    print("Creating behaviour tree")
    # Create and setup the tree
    bt = create_behavior_tree(robot)
    bt.setup(timeout=15)

    print("Starting Behavior Tree execution...")
    while robot.supervisor.step(robot.timestep) != -1:
        bt.tick()
        
        # Optional: Print status every few seconds
        if bt.root.status == py_trees.common.Status.SUCCESS:
            print("Task Completed Successfully!")
            break
        elif bt.root.status == py_trees.common.Status.FAILURE:
            print("Task Failed.")
            break

if __name__ == '__main__':
    main()