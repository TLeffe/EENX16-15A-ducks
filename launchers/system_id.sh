#!/bin/bash


source /environment.sh

dt-launchfile-init

rosrun my_package system_id_node.py

dt-launchfile-join

