#!/bin/bash
source /environment.sh
dt-launchfile.init
rosrun my_package komtest.py
dt-launchfile-join

