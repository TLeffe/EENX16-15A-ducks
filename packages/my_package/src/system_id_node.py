#!/usr/bin/env python3

import os
import math
import rospy
import csv
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped , WheelEncoderStamped
from std_msgs.msg import String
from sensor_msgs.msg import Imu



AXIS_LENGTH = 0.105          # meter — avstånd mellan hjulen
WHEEL_RADIUS = 0.035         # meter — hjulradius
WHEEL_CIRC = WHEEL_RADIUS * 2 * math.pi   # hjulets omkrets i meter
TICKS_PER_REV = 135          # ticks per varv

Accepted_angle = math.radians(30)
# -------------------------------------------------------
# PI-regulator för styrning 
# -------------------------------------------------------

OMEGA_MAX = 4.0              # max vridningshastighet (säkerhetsgräns
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
        self.imu_topic = f"/{self.vehicle_name}/imu_node/raw"
        self.VELOCITY = 0.3             # framåthastighet (m/s)
        self.DESIRED_THETA = 0.0          # önskad riktning (0 = rakt fram i radianer)
        
        self._ticks_left  = None
        self._ticks_right = None
        self._position = [0.0, 0.0, 0.0]          # Odometri — position (x, y, theta)
    
        self._v     = self.VELOCITY

        self._theta_error_integral = 0.0          # PI-state
        self._prev_theta_error = 0.0

        self.unwrapped_theta = 0
        self.senast_tid = rospy.get_time()
        self.calc_omega = 0
        self.latest_imu_gyro_z = 0.0

        self.gyro_bias = 0.0
        self.is_calibrating = False


        self._publisher = rospy.Publisher(self.twist_topic, Twist2DStamped, queue_size=1)    # Publisher för körkommandon
        self.sub_instructions = rospy.Subscriber(self.instruction_topic, String, self.callback_comm)
        self.sub_left = rospy.Subscriber(self.left_enc_topic,  WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self.right_enc_topic, WheelEncoderStamped, self.callback_right)
        self.imu_read = rospy.Subscriber (self.imu_topic, Imu, self.callback_imu)
        rospy.loginfo("Rak körning med PI-styrning startad")

        ##system id methods.
        self._csv_file = open('/data/angle_log.csv', 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(['timestamp', 'omega_to_spin', 'absolute_theta', 'calc_omega', 'gyroZ'])

    def _log_to_csv(self): #logga tid, desired angle, actual angle och theta error. 
        rospy.loginfo("skriver rad")
        time=rospy.get_time()
        self._csv_writer.writerow([time,self.DESIRED_THETA,self._position[2],self.calc_omega,self.latest_imu_gyro_z])

    def callback_imu(self, data):
        self.latest_imu_gyro_z = data.angular_velocity.z - self.gyro_bias

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

    def callback_left(self, data):
        self._ticks_left = data.data

    def callback_right(self, data):
        self._ticks_right = data.data
   
    def _publish_cmd(self, v, omega):       #  En hjälpfunktion som skickar körkommando till roboten
       self._publisher.publish(Twist2DStamped(v=v, omega=omega))       

    def run(self):
        rate = rospy.Rate(40)

        rospy.loginfo("Väntar på att sensorer vaknar....")
        while (self._ticks_left is None or self.latest_imu_gyro_z==0) and not rospy.is_shutdown():
            rate.sleep()
        self.calibrate_gyro(duration=2.0)
        rospy.loginfo(f"IMU bias = {self.gyro_bias}")
        prev_ticks_left  = self._ticks_left
        prev_ticks_right = self._ticks_right
        test_omega = 1.0 
        start_tid = rospy.get_time()
        rospy.loginfo("startar datainsamling...")
        while not rospy.is_shutdown():
            prev_ticks_left, prev_ticks_right = self.update_odometry(prev_ticks_left, prev_ticks_right)
            nuvarande_tid = rospy.get_time()
            passerad_tid = nuvarande_tid - start_tid
            rospy.loginfo(f"imutest Z:{self.latest_imu_gyro_z}")
            if passerad_tid < 2:
                v =0.2
                omega = 0
            elif passerad_tid < 5:
                v =0.2
                omega = test_omega
            elif passerad_tid < 8:
                v =0.2
                omega = 0
            elif passerad_tid < 11:
                v =0.2
                omega = -test_omega
            elif passerad_tid < 14:
                v =0.2
                omega = 0
            else:
                v = 0.0
                omega = 0
                self._publish_cmd(v,omega)
                rospy.loginfo("test färdigt")
                break

            self._publish_cmd(v,omega)
            self._csv_writer.writerow([nuvarande_tid, 
                                       omega, 
                                       self.unwrapped_theta,
                                       self.calc_omega,
                                       self.latest_imu_gyro_z])
            
            rate.sleep()


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
    node = TwistControlNode(node_name='system_id_node')
    node.run()
    rospy.spin()
