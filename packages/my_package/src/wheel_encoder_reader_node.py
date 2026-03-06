#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped

class WheelEncoderReaderNode(DTROS):

   def __init__(self, node_name):
       super(WheelEncoderReaderNode, self).__init__(
           node_name=node_name,
           node_type=NodeType.PERCEPTION
       )
       self._vehicle_name = os.environ['VEHICLE_NAME']
       self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"         # i den raden vi lyssnar på däcken
       self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"       #samm som oven 
       
       self._ticks_left = None
       self._ticks_right = None
       self._last_tricks_left = None        # Räkna antal meddelanden
       self._last_tricks_right = None

       self._left_msg_count = 0
       self._right_msg_count = 0

       rospy.loginfo(f"Lyssnar på vänstar encoder: {self._left_encoder_topic}")
       rospy.loginfo(f"Lyssnar på höger encoder : {self._right_encoder_topic}")

       self.sub_left = rospy.Subscriber(
           self._left_encoder_topic,
           WheelEncoderStamped,
           self.callback_left
       )
       self.sub_right = rospy.Subscriber(
           self._right_encoder_topic,
           WheelEncoderStamped,
           self.callback_right
       )

   def callback_left(self, data):
       self._left_msg_count += 1
       if self._left_msg_count ==1:
           rospy.loginfo(f"Vänter encoder aktiv! Resolution : {data.resolution} ticks /varv")
       self._last_tricks_left = self._ticks_left
       self._ticks_left = data.data

       if self._last_tricks_left is not None:
           delta = self._ticks_left- self._last_tricks_left
           if delta !=0:
               rospy.loginfo(f"Vänster hjul: {self._ticks_left} tick (andring: {delta:+d})")     # d= heltal, + visa alltid tecknet

      # rospy.loginfo_once(f"Left encoder resolution: {data.resolution}")
     #  rospy.loginfo_once(f"Left encoder type: {data.type}")
       #self._ticks_left = data.data

   def callback_right(self, data):
       self._right_msg_count +=1
       if self._right_msg_count ==1:
           rospy.loginfo(f"Höger encoder aktiv! Resolution: {data.resolution} ticks /varv ")
       self._last_tricks_right = self._ticks_right
       self._ticks_right = data.data

       if self._last_tricks_right is not None:
           delta = self._ticks_right - self._last_tricks_right
           if delta != 0:
               rospy.loginfo(f" Höger hjul : {self._ticks_right} ticks (andring: {delta:+d})")

       #rospy.loginfo_once(f"Right encoder resolution: {data.resolution}")
       #rospy.loginfo_once(f"Right encoder type: {data.type}")
       #self._ticks_right = data.data

   def run(self):
       rate = rospy.Rate(2)
       while not rospy.is_shutdown(): 
           if self._ticks_left is not None and self._ticks_right is not None:
               msg = (
                   f"Wheel encoder ticks [LEFT, RIGHT]: "
                   f"{self._ticks_left}, {self._ticks_right}"
               )
               rospy.loginfo(msg)
           rate.sleep()

if __name__ == '__main__':
   node = WheelEncoderReaderNode(node_name='wheel_encoder_reader_node')
   node.run()
   rospy.spin()
