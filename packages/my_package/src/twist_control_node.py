#!/usr/bin/env python3

import os
import math
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped , WheelEncoderStamped
from std_msgs.msg import String



AXIS_LENGTH = 0.105          # meter — avstånd mellan hjulen
WHEEL_RADIUS = 0.035         # meter — hjulradius
WHEEL_CIRC = WHEEL_RADIUS * 2 * math.pi   # hjulets omkrets i meter
TICKS_PER_REV = 135          # ticks per varv

Accepted_angle = math.radians(20)
# -------------------------------------------------------
# PI-regulator för styrning 
# -------------------------------------------------------
KP_THETA = 4.0               # proportionell — hur hårt vi styr mot rätt riktning
KI_THETA =  0.1           # integral — kompenserar konstant drift
OMEGA_MAX = 4.0              # max vridningshastighet (säkerhetsgräns)
KD_THETA= 0.2

class TwistControlNode(DTROS):


    def __init__(self, node_name):
        super(TwistControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.instruction = "" #Initiera en tom string för instruktionerna som tas emot
        self.prev_instructions = "" #initiera en tom string för för förra instruktionerna mottagna
        self.current_order_list = [] #tom lista som order förvaras i. 
        self.vehicle_name = os.environ['VEHICLE_NAME']
        self.instruction_topic = f"/{self.vehicle_name}/Comm_node/instructions"

        self.twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.left_enc_topic = f"/{self.vehicle_name}/left_wheel_encoder_driver_node/tick"
        self.right_enc_topic = f"/{self.vehicle_name}/right_wheel_encoder_driver_node/tick"
        self.VELOCITY = 0.3               # framåthastighet (m/s)
        self.DESIRED_THETA = 0.0          # önskad riktning (0 = rakt fram i radianer)
        self._ticks_left  = None
        self._ticks_right = None
        self._theta_error_integral = 0.0          # PI-state
        self._v     = self.VELOCITY
       
        self._position = [0.0, 0.0, 0.0]          # Odometri — position (x, y, theta)
        self._publisher = rospy.Publisher(self.twist_topic, Twist2DStamped, queue_size=1)    # Publisher för körkommandon
        self.sub_instructions = rospy.Subscriber(self.instruction_topic, String, self.callback_comm)
        self.sub_left = rospy.Subscriber(self.left_enc_topic,  WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self.right_enc_topic, WheelEncoderStamped, self.callback_right)
        rospy.loginfo("Rak körning med PI-styrning startad")
    def callback_comm(self,msg):
        self.instruction = msg.data
        rospy.loginfo(f"recieved instructions:{self.instruction}")

    def instruction_parse(self):  # tar hand om inkommande instruktioner, ska vara string på formen self.vehicle_name,x,y,theta,x1,y1,x2,y2 
        while not rospy.is_shutdown():
            if self.instruction != self.prev_instructions:
                self.prev_instructions = self.instruction
                self.current_order_list = self.prev_instructions.split(",") # gör om instruktionerna till en lista. 
                del self.current_order_list[0]  # ta bort namnet på roboten
                self._position[0:3] = self.current_order_list[0:3] #uppdatera postion och vinklar
                self.DESIRED_THETA = math.atan((self.current_order_list[0]-self.current_order_list[3])
                                               /self.current_order_list[1]-self.current_order_list[4])# beräkning önskadvinkel
            else:
                break # om inga nya instruktioner på topic, uppdatera inget

    def callback_left(self, data):
        self._ticks_left = data.data

    def callback_right(self, data):
        self._ticks_right = data.data
   
    def _publish_cmd(self, v, omega):       #  En hjälpfunktion som skickar körkommando till roboten
       self._publisher.publish(Twist2DStamped(v=v, omega=omega))       

    def check_angle_error(self):
        theta_error = self.DESIRED_THETA - self._position[2]
                #  Normalisera felet till intervallet [-pi, pi]
        while theta_error > math.pi:   
            theta_error -= 2 * math.pi

        while theta_error < -math.pi:
            theta_error += 2 * math.pi

        return theta_error
    
    def rotation_to_correct(self):
        rospy.loginfo(" Kolla vinkelfelet om det är större än 20 grader")
        while not rospy.is_shutdown():
            theta_error =  self.check_angle_error()

            if abs(theta_error) < Accepted_angle:
               
                break
            omega = KP_THETA * theta_error
            omega = max(-OMEGA_MAX, min(OMEGA_MAX, omega))     # roboten ska inte vrider sig för snabbt

            pass
    
    def run(self):
        rate = rospy.Rate(5)
        dt = 1.0 /5.0    # tidssteg i sekunder
        prev_ticks_left  = None
        prev_ticks_right = None
        rospy.loginfo("Väntar på encoder-data...")
        while (self._ticks_left is None or self._ticks_right is None) and not rospy.is_shutdown():
            rate.sleep() 

        prev_ticks_left  = self._ticks_left
        prev_ticks_right = self._ticks_right
        rospy.loginfo("Encoder-data mottagen — startar körning!")

        while not rospy.is_shutdown():
            self.instruction_parse()

            rospy.loginfo(f"recieved instructions:{self.instruction}")
            dNl = self._ticks_left  - prev_ticks_left
            dNr = self._ticks_right - prev_ticks_right

            dl = WHEEL_CIRC * (dNl / TICKS_PER_REV)   # vänster hjul i meter
            dr = WHEEL_CIRC * (dNr / TICKS_PER_REV)   # höger hjul i meter

            d =  (dl + dr) / 2.0                   # sträcka framåt
            dtheta = (dr - dl) / AXIS_LENGTH       # svängning i radianer eller förändning i vinkel

            # Midpoint-metoden — noggrannare positionsuppdatering
            midpoint_theta = self._position[2] + dtheta / 2.0      # Robotens vinkel mitt i rörelsen.
            self._position[0] += d * math.cos(midpoint_theta)      # Robotens x-position uppdateras.
            self._position[1] += d * math.sin(midpoint_theta)      # Robotens y-position uppdateras.
            self._position[2] += dtheta                            # Robotens rotation uppdateras

            prev_ticks_left  = self._ticks_left
            prev_ticks_right = self._ticks_right
            


            #   PI-STYRNING — håll roboten rakt (riktning robot ska ha - robots nuvarnde vinkel)
            theta_error = self.DESIRED_THETA - self._position[2]

                #  Normalisera felet till intervallet [-pi, pi]
            while theta_error > math.pi:   
                theta_error -= 2 * math.pi

            while theta_error < -math.pi:
               theta_error += 2 * math.pi
            
            self._prev_theta_error = theta_error
            self._theta_error_integral += theta_error * dt   # uppdatera integralen (I-delen)
            self._theta_error_integral = max(-3.0, min(3.0, self._theta_error_integral))
            self._derivatan = KD_THETA * (theta_error-self._prev_theta_error)/dt
            # PI-regulator      (omega = P-del + I-del)
            omega = KP_THETA * theta_error + KI_THETA * self._theta_error_integral + self._derivatan
            omega = max(-OMEGA_MAX, min(OMEGA_MAX, omega))     # roboten ska inte vrider sig för snabbt


            self._publish_cmd(v=self.VELOCITY, omega=omega)       # skicka kommando till roboten
            # rospy.loginfo(f" Skillanden {self._ticks_left - self._ticks_right}")
            # rospy.loginfo_throttle(
            #     1.0,  # en gång per sec
            #     f"Pos: x={self._position[0]:.3f}m  y={self._position[1]:.3f}m  "
            #     f"theta={math.degrees(self._position[2]):.1f}°  "
            #     f"fel={math.degrees(theta_error):.1f}°  omega={omega:.3f}"
            #    # f"skillnaden{self._ticks_left - self._ticks_right}"
            #     )
        
            rate.sleep()


    def on_shutdown(self):
        rospy.loginfo("Stoppar Roboten")
        stop = Twist2DStamped(v=0.0, omega=0.0)
        self._publisher.publish(stop)
        # try:
        #     stop = Twist2DStamped(v=0.0, omega=0.0)
        #     self._publisher.publish(stop)
        #     #rospy.sleep(0.5)
        # except:
        #     pass

if __name__ == '__main__':
    node = TwistControlNode(node_name='twist_control_node')
    node.run()
    rospy.spin()
