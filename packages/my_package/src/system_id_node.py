#!/usr/bin/env python3

import os
import math
import rospy
import csv
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped , WheelEncoderStamped
from std_msgs.msg import String



AXIS_LENGTH = 0.105          # meter — avstånd mellan hjulen
WHEEL_RADIUS = 0.035         # meter — hjulradius
WHEEL_CIRC = WHEEL_RADIUS * 2 * math.pi   # hjulets omkrets i meter
TICKS_PER_REV = 135          # ticks per varv

Accepted_angle = math.radians(30)
# -------------------------------------------------------
# PI-regulator för styrning 
# -------------------------------------------------------
KP_THETA = 0               # proportionell — hur hårt vi styr mot rätt riktning
KI_THETA =  0          # integral — kompenserar konstant drift
OMEGA_MAX = 4.0              # max vridningshastighet (säkerhetsgräns)
KD_THETA= 0
GOAL_THRESHOLD = 0.05       # 5 cm — mål nått
BASE_SPEED =  0.3
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
        self.VELOCITY = 0.3             # framåthastighet (m/s)
        self.DESIRED_THETA = 0.0          # önskad riktning (0 = rakt fram i radianer)
        
        self._ticks_left  = None
        self._ticks_right = None
        self._position = [0.0, 0.0, 0.0]          # Odometri — position (x, y, theta)

        self.goal_pose = [4, 0]               # (x, y) —>>> målet vi kör mot
        self._v     = self.VELOCITY

        self._theta_error_integral = 0.0          # PI-state
        self._prev_theta_error = 0.0

        self._publisher = rospy.Publisher(self.twist_topic, Twist2DStamped, queue_size=1)    # Publisher för körkommandon
        self.sub_instructions = rospy.Subscriber(self.instruction_topic, String, self.callback_comm)
        self.sub_left = rospy.Subscriber(self.left_enc_topic,  WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self.right_enc_topic, WheelEncoderStamped, self.callback_right)
        rospy.loginfo("Rak körning med PI-styrning startad")

        ##system id methods.
        self._csv_file = open('/data/angle_log.csv', 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(['timestamp', 'desired_theta_deg', 'actual_theta_deg'])

    def _log_to_csv(self): #logga tid, desired angle, actual angle och theta error. 
        rospy.loginfo("skriver rad")
        time=rospy.get_time()
        self._csv_writer.writerow([time,self.DESIRED_THETA,self._position[2]])


    def callback_comm(self,msg):
        self.instruction = msg.data
        rospy.loginfo(f"recieved instructions:{self.instruction}")
    


    def instruction_parse(self):  # tar hand om inkommande instruktioner, ska vara string på formen self.vehicle_name,x,y,theta,x1,y1,x2,y2 
        while not rospy.is_shutdown():
            if self.instruction != self.prev_instructions:
                self.prev_instructions = self.instruction
                self.current_order_list = self.prev_instructions.split(",") # gör om instruktionerna till en lista. 
                del self.current_order_list[0] # ta bort namnet på roboten
                self.current_order_list = [float(i) for i in self.current_order_list]
                self._position[0:3] = self.current_order_list[0:3] #uppdatera postion och vinklar
                self.goal_pose[0:2] = self.current_order_list[3:5]
            else:
                break # om inga nya instruktioner på topic, uppdatera inget

    def callback_left(self, data):
        self._ticks_left = data.data

    def callback_right(self, data):
        self._ticks_right = data.data
   
    def _publish_cmd(self, v, omega):       #  En hjälpfunktion som skickar körkommando till roboten
       self._publisher.publish(Twist2DStamped(v=v, omega=omega))       

    def normalize_angle(self,angle):       #  Normalisera felet till intervallet [-pi, pi]
        while angle > math.pi:   
            angle -= 2 * math.pi

        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def reset_PID(self):             # nollställa PID, anropas denna fkt när roboten ska rotera
        self._theta_error_integral = 0.0
        self._prev_theta_error     = 0.0
        
    def check_angle_error(self):
        theta_error = self.DESIRED_THETA - self._position[2]
        return self.normalize_angle(theta_error)
    
    def PID_omega(self, theta_error,dt):
        self._derivatan = KD_THETA * (theta_error-self._prev_theta_error)/dt
        self._prev_theta_error = theta_error

        self._theta_error_integral += theta_error * dt   # uppdatera integralen (I-delen)
        self._theta_error_integral = max(-8.0, min(8.0, self._theta_error_integral))

        omega = KP_THETA * theta_error + KI_THETA * self._theta_error_integral + self._derivatan
        omega = max(-OMEGA_MAX, min(OMEGA_MAX, omega))     # roboten ska inte vrider sig för snabbt

        return omega

    
    def rotation_to_correct(self, rate, dt, prev_ticks_left, prev_ticks_right):
        rospy.loginfo(
            f"Vinkelfel > 20 grader —>>> roterar pa plats "
            f"Nuvarande: {math.degrees(self._position[2]):.1f} grader"
            f"Önskar: {math.degrees(self.DESIRED_THETA):.1f} grader"
        )
        self.reset_PID()
        while not rospy.is_shutdown():
            self.instruction_parse()
            prev_ticks_left, prev_ticks_right = self.update_odometry(
                prev_ticks_left, prev_ticks_right
            )
            theta_error =  self.check_angle_error()
            if abs(theta_error) < Accepted_angle:
                rospy.loginfo(f"Vinkle OK!  fel={math.degrees(theta_error):1f} grader")
                break
            
            omega = self.PID_omega(theta_error, dt)
            self._publish_cmd (v=0.0, omega=omega)      # V = 0 stå still under rotation   

            rospy.loginfo_throttle(
                1, 
                f"Roterar fel {math.degrees(theta_error):.1f} grader omega = {omega:.3f}"
            )
            rate.sleep()
        self.reset_PID()
        return prev_ticks_left, prev_ticks_right

    def straight_forward(self, dt): 
        #   kör framåt med PID mot önskade theta när vinkelfel < 20 grader.
        theta_error = self.check_angle_error()  
        omega = self.PID_omega(theta_error, dt)
        self._publish_cmd(v=self.VELOCITY, omega= omega)

    def Goal_reached(self):
        """Returnera True om Roboten är inom 5 cm från målet"""
        if self.goal_pose is None:
            return False
        dx = self.goal_pose[0] - self._position[0]
        dy = self.goal_pose[1] - self._position[1]

        return math.sqrt(dx**2 + dy**2) < GOAL_THRESHOLD

    def update_odometry(self, prev_tick_left, prev_ticks_right):

        dNl = self._ticks_left  - prev_tick_left
        dNr = self._ticks_right - prev_ticks_right
        dl = WHEEL_CIRC * (dNl / TICKS_PER_REV)   # vänster hjul i meter
        dr = WHEEL_CIRC * (dNr / TICKS_PER_REV)   # höger hjul i meter
        d =  (dl + dr) / 2.0                   # sträcka framåt
        dtheta = (dr - dl) / AXIS_LENGTH       # svängning i radianer eller förändning i vinkel
        midpoint_theta  = self._position[2] + dtheta / 2.0
        self._position[0] += d * math.cos(midpoint_theta)
        self._position[1] += d * math.sin(midpoint_theta)
        self._position[2]  = self.normalize_angle(self._position[2] + dtheta)   # Utan normalisering kan roboten få problem när man beräknar rotationsfel

        return self._ticks_left, self._ticks_right

    def calculate_desired_direction(self):

        dx_g = self.goal_pose[0] - self._position[0]        # beräknar skillnaden mellan målet och nuvarande pos
        dy_g = self.goal_pose[1] - self._position[1]

        dist_goal = math.sqrt(dx_g**2 + dy_g**2)
        
        if dist_goal < 0.005:       # Om roboten redan är vid målet, avsluta funktionen
            return
        # self.DESIRED_THETA = math.atan2(dy_g, dx_g)     # Beräknar önskad vinkel (theta) mot målet
        self.DESIRED_THETA = math.atan2(self.goal_pose[1] - self._position[1],self.goal_pose[0] - self._position[0]  )  
        self.VELOCITY  = BASE_SPEED
 


    def run(self):
        rate = rospy.Rate(25)
        dt = 1.0 /25.0    # tidssteg i sekunder

        rospy.loginfo("Väntar på encoder data.")
        while (self._ticks_left is None or self._ticks_right is None) and not rospy.is_shutdown():
            rate.sleep()         
        prev_ticks_left  = self._ticks_left
        prev_ticks_right = self._ticks_right
        rospy.loginfo(f"Encoder OK! start: V= {prev_ticks_left}  H={prev_ticks_right}")
        rospy.loginfo(f"Start -->>> Mal:{self.goal_pose}")
        test_omega = 2.0
        start_tid = rospy.get_time()
        test_tid = 5.0

        while not rospy.is_shutdown():
            # Uppdaterar odometri
            prev_ticks_left, prev_ticks_right = self.update_odometry(prev_ticks_left, prev_ticks_right)
            nuvarande_tid = rospy.get_time()
            passerad_tid = nuvarande_tid - start_tid
            
            if passerad_tid < test_tid:
                v =0.2
                omega = test_omega
            else:
                v = 0.0
                omega = 0
                self._publish_cmd(v,omega)
                rospy.loginfo("test färdigt")
                break

            self._publish_cmd(v,omega)
            self._csv_writer.writerow([nuvarande_tid, omega, self._position[2]])
            
            rate.sleep()

            
            # self.calculate_desired_direction()       #  Bräkna önskade  riktning 
            # theta_error = self.check_angle_error()    # Kolla vinkelfel och styr

            # # if abs(theta_error) > Accepted_angle:
            # #     # Fel > 20 grader — stanna och rotera på plats
            # #     self._publish_cmd(v=0.0, omega= 0.0)
            # #     prev_ticks_left,prev_ticks_right = self.rotation_to_correct(rate, dt, prev_ticks_left, prev_ticks_right)
            # # else:
            # self.straight_forward(dt)     # Fel < 20 grader — kör rakt med PID


            # rate.sleep()

    def on_shutdown(self):
        rospy.loginfo("Stoppar Roboten")
        stop = Twist2DStamped(v=0.0, omega=0.0)
        self._publisher.publish(stop)
        self._csv_file.close()  
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
