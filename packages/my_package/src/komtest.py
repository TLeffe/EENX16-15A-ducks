#!/usr/bin/env python3


import os
import rospy
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType
import socket
PORT = 8765


class komtest(DTROS):
   def __init__(self, node_name):
       super(komtest, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
       self._vehicle_name = os.environ['VEHICLE_NAME']
       self._publisher = rospy.Publisher('chatter', String, queue_size=10)


   def run(self):
        rate = rospy.Rate(1)  # 1 Hz
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as client:
            client.setsockopt (socket.SOL_SOCKET,socket.SO_REUSEADDR, 1)
            client.bind(('',PORT))
            while not rospy.is_shutdown():
                data, addr = client.recvfrom(1024) 
                message=data.decode()
                rospy.loginfo(f"hearing:'{message}'")
                self._publisher.publish(message)
                rate.sleep()


if __name__ == '__main__':
   node = komtest(node_name='komtest')
   node.run()
   rospy.spin()