#!/bin/bash


source /environment.sh


dt-launchfile-init
rosrun my_package twist_control_node.py & 
rosrun my_package obstacle_detection_node.py
dt-launchfile-join
