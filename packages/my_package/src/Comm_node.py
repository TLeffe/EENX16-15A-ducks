#!/usr/bin/env python3


import os
import rospy
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType
import socket
PORT = 8765


class Comm(DTROS):
   def __init__(self, node_name):
       super(Comm, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
       self._vehicle_name = os.environ['VEHICLE_NAME']
       self.instruction_topic = f"/{self._vehicle_name}/Comm_node/instructions"
       self._publisher = rospy.Publisher('instructions', String, queue_size=10) # ut topic


   def run(self):
        while not rospy.is_shutdown():
            rate = rospy.Rate(1)  # 10 Hz, how often we check for new messages
            with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as client:
                client.setsockopt (socket.SOL_SOCKET,socket.SO_REUSEADDR, 1)
                client.bind(('',PORT))    # Lyssnar på PORT som definerats tidigare.   
                while not rospy.is_shutdown():
                    data = client.recv(1024) #begränsar storleken på mottaget paket. 
                    message=data.decode()
                    rospy.loginfo(f"hearing:'{message}'") #skickar vad vi tar emot i terminalen, debugging
                    #todo, parse vad vi vill höra från meddelandet.
                    start_of_relevance = message.find(self._vehicle_name) #find start string of relevant data  determined with the bot name
                    rospy.loginfo(f"start: '{start_of_relevance}'")
                    end_of_relevance = message.find("nd", start_of_relevance) # find end of string with nd as marker
                    rospy.loginfo(f"end: '{end_of_relevance}'")
                    rospy.loginfo(f"star: '{start_of_relevance}' end: {end_of_relevance}")
                    relevant = message[start_of_relevance:end_of_relevance]
                    rospy.loginfo(f"hearing again:'{message}'")
                    # relevant = message.split(",")
                    # del relevant[0]
                    rospy.loginfo(f"hearing again:'{relevant}'")
                    self._publisher.publish(relevant) # publiserar på topic incoming_data
                    rate.sleep()


if __name__ == '__main__':
   node = Comm(node_name='Comm_node')
   node.run()
   rospy.spin()