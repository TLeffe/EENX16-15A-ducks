#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped

# throttle and direction for each wheel
THROTTLE_LEFT = 0.5        # 50% throttle
DIRECTION_LEFT = 1         # forward
THROTTLE_RIGHT = 0.5       # 30% throttle
DIRECTION_RIGHT = 1       # backward

class WheelControlNode(DTROS):

   def __init__(self, node_name):
       super(WheelControlNode, self).__init__(
           node_name=node_name,
           node_type=NodeType.GENERIC
       )
       
       vehicle_name = os.environ['VEHICLE_NAME']
       wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
       self._vel_left = THROTTLE_LEFT * DIRECTION_LEFT
       self._vel_right = THROTTLE_RIGHT * DIRECTION_RIGHT

       self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

       # skriva lite info här som vi kan se 
       rospy.loginfo("====== Rak framkörning konfigererad==========")

       rospy.loginfo(f"Vänster hjul : {self._vel_left: .2f} (Thorttle: {THROTTLE_LEFT}, Rktning: {DIRECTION_LEFT})")
       rospy.loginfo(f"Höger hjul : {self._vel_right: .2f} (Thorttle: {THROTTLE_RIGHT}, Rktning: {DIRECTION_RIGHT})")
       if self._vel_left == self._vel_right:
            rospy.loginfo("Ja, hjulen är synkade")
       else:
            rospy.loginfo("Nej, hjulen är inte synkade")

   def run(self):
       rate = rospy.Rate(10)    # Har ändrat från 0.1 till 10 HZ
       message = WheelsCmdStamped(
        vel_left=self._vel_left,
        vel_right=self._vel_right
        )
       counter = 0          # vill räkna  
       while not rospy.is_shutdown():
           self._publisher.publish(message)
           rate.sleep()
           counter +=1    # räknar
           if counter % 100 ==0 :       #När counter är delbart med 100
               rospy.loginf(f"Fortfarande i rörelse: Vänster={self._vel_left:.2f}, Höger={self._vel_right:.2f} ")
               
               counter = 0    #Så den börjar räkna om igen mot nästa 100.
           rate.sleep()
       

   def on_shutdown(self):
       stop = WheelsCmdStamped(vel_left=0, vel_right=0)
       self._publisher.publish(stop)

if __name__ == '__main__':
   node = WheelControlNode(node_name='wheel_control_node')
   node.run()
   rospy.spin()
