#!/usr/bin/env python3

import os
import math
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped , WheelEncoderStamped
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import Imu



AXIS_LENGTH = 0.105          # meter — avstånd mellan hjulen
WHEEL_RADIUS = 0.035         # meter — hjulradius
WHEEL_CIRC = WHEEL_RADIUS * 2 * math.pi   # hjulets omkrets i meter
TICKS_PER_REV = 135          # ticks per varv

Accepted_angle = math.radians(10)
# -------------------------------------------------------
# PI-regulator för styrning 
# -------------------------------------------------------
KP_THETA = 8               # proportionell — hur hårt vi styr mot rätt riktning
KI_THETA =  0.2          # integral — kompenserar konstant drift
OMEGA_MAX = 0.5           # max vridningshastighet (säkerhetsgräns)
KD_THETA= 0
GOAL_THRESHOLD = 0.05       # 5 cm — mål nått
BASE_SPEED =  0.5
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
        
        self.desired_theta_topic = f"/{self.vehicle_name}/twist_control_node/desired_theta"
        self.current_theta_topic = f"/{self.vehicle_name}/twist_control_node/current_theta"
        self.obstacle_topic      = f"/{self.vehicle_name}/obstacle_detection_node/obstacle_detected"
        if self.vehicle_name == 'duck4':
            KP_THETA = 22
            KI_THETA = 0.01
            KD_THETA = 0.1
        elif self.vehicle_name == 'duck3':
            KP_THETA = 32
            KI_THETA = 0.1
            KD_THETA = 0.1
        elif self.vehicle_name == 'duck6':
            KP_THETA = 12
            KI_THETA = 0.1
            KD_THETA = 0.1
        #----------Things for IMU----------#
        self.imu_topic = f"/{self.vehicle_name}/imu_node/raw"
        self.senast_tid = rospy.get_time()
        self.calc_omega = 0
        self.gyro_bias = 0.0
        self.imu_recieved = False
        self.latest_imu_gyro_z = 0
        self.imu_read = rospy.Subscriber (self.imu_topic, Imu, self.callback_imu)
        #-------------------------------------------------------------------------#

        self.VELOCITY = 0.3              # framåthastighet (m/s)
        self.DESIRED_THETA = 0.0          # önskad riktning (0 = rakt fram i radianer)
        
        self._ticks_left  = None
        self._ticks_right = None
        self._position = [0.0, 0.0, 0.0]          # Odometri — position (x, y, theta)

        self.goal_pose = [0, 0]               # (x, y) —>>> målet vi kör mot
   

        self._theta_error_integral = 0.0          # PI-state
        self._prev_theta_error = 0.0
        self.obstacle_active = False     #True när obstacle_detection håller på att undvika

        self._publisher = rospy.Publisher(self.twist_topic, Twist2DStamped, queue_size=1)    # Publisher för körkommandon
   
        self.desired_theta_pub = rospy.Publisher(self.desired_theta_topic, Float32, queue_size=1)
        self.current_theta_pub = rospy.Publisher(self.current_theta_topic, Float32, queue_size=1)

        self.sub_instructions = rospy.Subscriber(self.instruction_topic, String, self.callback_comm)
        self.sub_left = rospy.Subscriber(self.left_enc_topic,  WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self.right_enc_topic, WheelEncoderStamped, self.callback_right)

        self.sub_obstacle = rospy.Subscriber(self.obstacle_topic, Bool, self.callback_obstacle)
        rospy.loginfo("Rak körning med PI-styrning startad")
    
    def callback_comm(self,msg):
        self.instruction = msg.data
        rospy.loginfo(f"recieved instructions:{self.instruction}")
    
    def callback_imu(self, data):
        self.imu_recieved = True
        self.latest_imu_gyro_z = data.angular_velocity.z


    def calibrate_gyro(self, duration=2.0):
        rospy.loginfo("Starting Gyro Calibration...")
        
        samples = []
        start_time = rospy.get_time()
        
        # Use a high-frequency loop to grab as many samples as possible
        rate = rospy.Rate(50) 
        while rospy.get_time() - start_time < duration and not rospy.is_shutdown():
            # latest_imu_gyro_z is updated in the callback
            samples.append(self.latest_imu_gyro_z)
            rate.sleep()

        if len(samples) > 0:
            self.gyro_bias = sum(samples) / len(samples)
            rospy.loginfo(f"Calibration Complete. Bias: {self.gyro_bias:.5f} rad/s")
        else:
            rospy.logwarn("Calibration failed: No IMU samples received.") 

    
    
    def callback_obstacle(self, msg):  # Hanterar hinderstatus
        self.obstacle_active = msg.data
        if self.obstacle_active:
            rospy.loginfo(f"TwistControl: Hinder aktivt ------> pausar körning")
        #    self._publish_cmd(v =0.0, omega= 0.0)
        else:
            rospy.loginfo(f"TwistControl: Hinder klart ------> återupptar körning")

    def instruction_parse(self):  # tar hand om inkommande instruktioner, ska vara string på formen self.vehicle_name,x,y,theta,x1,y1,x2,y2 
        while not rospy.is_shutdown():
            if self.instruction != self.prev_instructions:
                self.prev_instructions = self.instruction
                self.current_order_list = self.prev_instructions.split(",") # gör om instruktionerna till en lista. 
                del self.current_order_list[0] # ta bort namnet på roboten
                self.current_order_list = [float(i) for i in self.current_order_list]
                self._position[0:3] = self.current_order_list[0:3] #uppdatera postion och vinklar
                self._position[2] = self._position[2]
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
        self._theta_error_integral = max(-5.0, min(5.0, self._theta_error_integral))

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
            if self.obstacle_active:
                rospy.loginfo_throttle(1.0, "rotation_to_correct: hinder aktivt, pausar rotation")
                dt, prev_ticks_left, prev_ticks_right = self.update_odometry(prev_ticks_left, prev_ticks_right)
                rate.sleep()
                continue
            self.instruction_parse()
            dt, prev_ticks_left, prev_ticks_right = self.update_odometry(
                prev_ticks_left, prev_ticks_right
            )
            theta_error =  self.check_angle_error()
            if abs(theta_error) < Accepted_angle:
                rospy.loginfo(f"Vinkle OK!  fel={math.degrees(theta_error):1f} grader")
                break
            
            omega = self.PID_omega(theta_error, dt)
            self._publish_cmd (v=0.00, omega=omega)      # V = 0 stå still under rotation   

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
        # rospy.loginfo(f"nu ska jag publicerat")

    def Goal_reached(self):
        """Returnera True om Roboten är inom 5 cm från målet"""
        if self.goal_pose is None:
            return False
        dx = self.goal_pose[0] - self._position[0]
        dy = self.goal_pose[1] - self._position[1]

        return math.sqrt(dx**2 + dy**2) < GOAL_THRESHOLD

    def update_odometry(self, prev_tick_left, prev_ticks_right):
        current_time = rospy.get_time()
        dt = current_time - self.senast_tid
        # if dt <= 0:
        #     dt = 1/25

        dNl = self._ticks_left  - prev_tick_left
        dNr = self._ticks_right - prev_ticks_right

        dl = WHEEL_CIRC * (dNl / TICKS_PER_REV)   # vänster hjul i meter
        dr = WHEEL_CIRC * (dNr / TICKS_PER_REV)   # höger hjul i meter
        d =  (dl + dr) / 2.0                   # sträcka framåt
        dtheta_enc = (dr - dl) / AXIS_LENGTH       # svängning i radianer eller förändning i vinkel
        alpha = 0.85
        gyro_dtheta = (self.latest_imu_gyro_z-self.gyro_bias)*dt
        fused_dtheta = alpha *gyro_dtheta + (1-alpha) * dtheta_enc # använder både gyro och enc för rotation. 

        midpoint_theta  = self._position[2] + fused_dtheta / 2.0
        self._position[0] += d * math.cos(midpoint_theta)
        self._position[1] += d * math.sin(midpoint_theta)
        self._position[2]  = self.normalize_angle(self._position[2] + fused_dtheta)   # Utan normalisering kan roboten få problem när man beräknar rotationsfel
        self.senast_tid = current_time
        return dt, self._ticks_left, self._ticks_right

    def calculate_desired_direction(self):

    #     dx_g = self.goal_pose[0] - self._position[0]        # beräknar skillnaden mellan målet och nuvarande pos
    #     dy_g = self.goal_pose[1] - self._position[1]

    #     dist_goal = math.sqrt(dx_g**2 + dy_g**2)
        
    #     if dist_goal < 0.005:       # Om roboten redan är vid målet, avsluta funktionen
    #         return
    #     # self.DESIRED_THETA = math.atan2(dy_g, dx_g)     # Beräknar önskad vinkel (theta) mot målet
    #     self.DESIRED_THETA = math.atan2(self.goal_pose[1] - self._position[1],self.goal_pose[0] - self._position[0]  )  
    #     dist = math.sqrt(dx_g**2 + dy_g**2)
    #     self.VELOCITY  = min(BASE_SPEED,dist)

    #     # publicera desired_theta och current theta varje cykel, då obstacle_detection läser dessa info
    #     self.desired_theta_pub.publish(Float32(data=self.DESIRED_THETA))
    #    # self.current_theta_pub.publish(Float32(data=self._position[2]))
    
        dx_g = self.goal_pose[0] - self._position[0]        # beräknar skillnaden mellan målet och nuvarande pos
        dy_g = self.goal_pose[1] - self._position[1]

        dist_goal = math.sqrt(dx_g**2 + dy_g**2)
        phi_goal = math.atan2(dy_g,dx_g) #heading to goal
        
        if dist_goal < 0.05:       # Om roboten redan är vid målet, avsluta funktionen
            return
        k_lateral = 1.5
        #lateral_correction = -k_lateral * self._position[1]
        # self.DESIRED_THETA = math.atan2(dy_g, dx_g)     # Beräknar önskad vinkel (theta) mot målet
        self.DESIRED_THETA = phi_goal
        dist = math.sqrt(dx_g**2 + dy_g**2)
        self.VELOCITY  = min(BASE_SPEED,dist)

        # publicera desired_theta och current theta varje cykel, då obstacle_detection läser dessa info
        self.desired_theta_pub.publish(Float32(data=self.DESIRED_THETA))
       # self.current_theta_pub.publish(Float32(data=self._position[2]))

    def run(self):
        rate = rospy.Rate(25)
        


#        rospy.loginfo("Väntar på Startposition from Comm-node.")
#        while self.position is  None and not rospy.is_shutdown():
#            self.instruction_parse()
 #           rospy.loginfo_throttle(2, "Väntar på init_pose...")
  #          rate.sleep()
        
        rospy.loginfo("Väntar på encoder data.")
        while (self._ticks_left is None or self._ticks_right is None) and not rospy.is_shutdown():
            rate.sleep() 

        if rospy.is_shutdown():
            return
        self.calibrate_gyro(duration=2.0)
        
        prev_ticks_left  = self._ticks_left
        prev_ticks_right = self._ticks_right
        self.senast_tid = rospy.get_time()
        rospy.loginfo(f"Encoder OK! start: V= {prev_ticks_left}  H={prev_ticks_right}")
        rospy.loginfo(f"Start -->>> Mal:{self.goal_pose}")

        while not rospy.is_shutdown():
            self.instruction_parse()
            dt, prev_ticks_left, prev_ticks_right = self.update_odometry(prev_ticks_left, prev_ticks_right)
            self.current_theta_pub.publish(Float32(data=self._position[2]))
            if self.obstacle_active:
                self._publish_cmd(v=0.0, omega=0.0)
                rate.sleep()
                continue

            # Uppdaterar odometri

            if self.Goal_reached():
                rospy.loginfo_throttle(2, "Är i önskade position  :)")
                self._publish_cmd(v=0.0, omega=0.0)
                rate.sleep()
                continue

            self.calculate_desired_direction()       #  Bräkna önskade  riktning 
            theta_error = self.check_angle_error()    # Kolla vinkelfel och styr

            if abs(theta_error) > Accepted_angle:
                # Fel > 20 grader — stanna och rotera på plats
                self._publish_cmd(v=0.0, omega= 0.0)
                prev_ticks_left,prev_ticks_right = self.rotation_to_correct(rate, dt, prev_ticks_left, prev_ticks_right)
            else:
                self.straight_forward(dt)     # Fel < 20 grader — kör rakt med PID

            ###### hur får vi positionen?  vilket topic mocap skickar på?##############################

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
