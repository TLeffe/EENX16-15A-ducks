import os
import cv2       #  används för att bearbeta kamerabilder
import math
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import Range, CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge


TOF_THRESHOLD   = 0.5    # meter — triggar kameran om föremål är närmare än detta
AVOID_OMEGA     = 2.0    #  hur hårt roboten svänger för att undvika (rad/s )
AVOID_VELOCITY  = 0.2    #  hastighet under undvikande
AVOID_DURATION  = 1.5    # hur länge roboten svänger (sek)

 
class ObstacleDetectionNode(DTROS):

    def __init__(self, node_name):
        super(ObstacleDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']

        self.twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.tof_topic = f"/{self.vehicle_name}/front_center_tof_driver_node/range"
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"


        self.tof_range = float ('inf')     # Senaste avståndet från ToF
        self.camera_active = False         # True när kameran analyserar
        self.obstacle_detected  = False

    def callback_tof(self, msg):   #  Körs varje gång ToF-sensorn skickar ett nytt avstånd.

        self.tof_range = msg.range
        if msg < TOF_THRESHOLD:         # Kolla om avståndet är mindre än tröskelvärdet
            if not self.camera_active:           # Om kameran inte redan är aktiv
                rospy.loginfo(f"TOF: Formål pa {msg.range:.2f}m --> aktivera kamera")
                self.camera_active = True
        else:
            if self.camera_active:             # Om avståndet är större än tröskeln
                rospy.loginfo(f"ToF: Vagen klar ({msg.range:.2f}m) — kamera inaktiv")
                self.camera_active = False     # Inget hinder nära --> stäng av kamera-analys
                self.obstacle_detected = False

    def callback_camera(self, msg):     #   Körs varje gång kameran skickar en ny bild.

        if not self.camera_active:      # Hoppa över om kameran inte är aktiv
            return

        pass

    pass