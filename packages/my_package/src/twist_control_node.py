#!/usr/bin/env python3
"""
Systemöversikt:
    1. Tar emot målkoordinater (x, y) via Comm_node.
    2. Beräknar önskad körriktning mot målet med atan2.
    3. Reglerar vinkeln med en PID-regulator baserad på odometri.
    4. Pausar körningen när obstacle_detection_node rapporterar ett aktivt hinder.
    5. Stannar när roboten är inom GOAL_THRESHOLD från målet.
"""
import os
import math
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped , WheelEncoderStamped
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import Imu


# ============================
# Robotgeometri
# ============================
AXIS_LENGTH = 0.105                       # [m] Avstånd mellan hjulen
WHEEL_RADIUS = 0.035                      # [m] Hjulradius
WHEEL_CIRC = WHEEL_RADIUS * 2 * math.pi   # [m] Hjulets omkrets
TICKS_PER_REV = 135                       # Encoder-ticks per varv på hjulen

# ============================
# PID-regulator för styrning 
# ============================
Accepted_angle = math.radians(10)         # [rad] Max tillåtet vinkelfel för rak körning

OMEGA_MAX = 0.5                           # [rad/sec] Max vridningshastighet för säkerhet

# ============================
# Körparametrar
# ============================
GOAL_THRESHOLD = 0.05                     # [m] Max avvikelse från mål för att räknas som "nått mål"
BASE_SPEED =  0.5                         # [m/s] Max framåthastighet

class TwistControlNode(DTROS):
    """
    ROS-nod för rörelseReglering på Duckiebot.
 
    Kombinerar hjulenkodrar och IMU-gyro i en fusionerad odometri,
    och styr roboten mot ett målkoordinat med PID-vinkelreglering.
    """

    def __init__(self, node_name):
        super(TwistControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self.vehicle_name = os.environ['VEHICLE_NAME']

        # Topics
        self.instruction_topic = f"/{self.vehicle_name}/Comm_node/instructions"
        self.twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.left_enc_topic = f"/{self.vehicle_name}/left_wheel_encoder_driver_node/tick"
        self.right_enc_topic = f"/{self.vehicle_name}/right_wheel_encoder_driver_node/tick"
        self.desired_theta_topic = f"/{self.vehicle_name}/twist_control_node/desired_theta"
        self.current_theta_topic = f"/{self.vehicle_name}/twist_control_node/current_theta"
        self.obstacle_topic      = f"/{self.vehicle_name}/obstacle_detection_node/obstacle_detected"
        
        # Olika robotar har olika motoregenskaper och kräver individuell inställning.
        self.KP_THETA = 1                              # Proportionell del- hur aggressivt roboten styr mot önskad vinkel
        self.KI_THETA = 1                           # Integraldel- kompenserar för konstant drift
        self.KD_THETA= 1                               # Derivatdel - dämpning av snabb vinkeländring
        # här stoppar ni in era beräknade PID parametrar för varje individuell robot
        if self.vehicle_name == 'placeholdername1':
            self.KP_THETA = 0
            self.KI_THETA = 0
            self.KD_THETA = 0
        elif self.vehicle_name == 'placeholdername2':
            self.KP_THETA = 0
            self.KI_THETA = 0
            self.KD_THETA = 0
        elif self.vehicle_name == 'placeholdername3':
            self.KP_THETA = 0
            self.KI_THETA = 0
            self.KD_THETA = 0
        else:
            pass

        # Instruktionshantering
        self.instruction = ""                 # Senast mottagen instruktionssträng
        self.prev_instructions = ""           # Föregående instruktionssträng (för förändringsdetektion)
        self.current_order_list = []          # Instruktioner parsade till en lista med floats

        # IMU-variabler 
        self.imu_topic = f"/{self.vehicle_name}/imu_node/raw"
        self.senast_tid = rospy.get_time()
        self.calc_omega = 0
        self.gyro_bias = 0.0                 # Beräknat biasvärde från kalibrering
        self.imu_recieved = False            # True efter första IMU-mätning
        self.latest_imu_gyro_z = 0           # Senaste gyromatning kring z-axeln [rad/s]
        
        self.imu_read = rospy.Subscriber (self.imu_topic, Imu, self.callback_imu)
        
        # Körvariabler
        self.VELOCITY = 0.3                  # [m/s] Aktuell körhastighet
        self.DESIRED_THETA = 0.0             # [rad] Önskad körriktning
        self._position = [0.0, 0.0, 0.0]     # Odometriposition [x, y, theta]
        self.goal_pose = [0, 0]              # Målkoordinat [x, y]
        self.obstacle_active = False         # True när hindernoden pausar körningen

        # Enkodertillstånd
        self._ticks_left  = None
        self._ticks_right = None

   
        # PID-tillstånd 
        self._theta_error_integral = 0.0        
        self._prev_theta_error     = 0.0

        # Publishers
        self._publisher = rospy.Publisher(self.twist_topic, Twist2DStamped, queue_size=1)
        self.desired_theta_pub = rospy.Publisher(self.desired_theta_topic, Float32, queue_size=1)
        self.current_theta_pub = rospy.Publisher(self.current_theta_topic, Float32, queue_size=1)

        # Subscribers
        self.sub_instructions = rospy.Subscriber(self.instruction_topic, String, self.callback_comm)
        self.sub_left = rospy.Subscriber(self.left_enc_topic,  WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self.right_enc_topic, WheelEncoderStamped, self.callback_right)
        self.sub_obstacle = rospy.Subscriber(self.obstacle_topic, Bool, self.callback_obstacle)
        
        rospy.loginfo("Rak körning med PI-styrning startad")
    
    def callback_comm(self,msg):
        """Tar emot instruktionssträng från Comm_node."""
        self.instruction = msg.data
        rospy.loginfo(f"recieved instructions:{self.instruction}")
    
    def callback_imu(self, data):
        """Tar emot rådata från IMU och sparar gyromatningen kring z-axeln."""
        self.imu_recieved = True
        self.latest_imu_gyro_z = data.angular_velocity.z


    def calibrate_gyro(self, duration=2.0):
        """
        Samlar in gyromatningar under 'duration' sekunder för att beräkna bias.
        Args:
            duration: Kalibreringtid i sekunder.
        """
        rospy.loginfo("Starting Gyro Calibration...")
        
        samples = []
        start_time = rospy.get_time()        
        rate = rospy.Rate(50) 

        while rospy.get_time() - start_time < duration and not rospy.is_shutdown():
            samples.append(self.latest_imu_gyro_z)
            rate.sleep()

        if len(samples) > 0:
            self.gyro_bias = sum(samples) / len(samples)
            rospy.loginfo(f"Calibration Complete. Bias: {self.gyro_bias:.5f} rad/s")
        else:
            rospy.logwarn("Calibration failed: No IMU samples received.") 

    
    
    def callback_obstacle(self, msg):
        """
        Hanterar hinderstatus från obstacle_detection_node.
        Sätter obstacle_active = True när ett hinder undviks, False annars.
        """
        self.obstacle_active = msg.data
        if self.obstacle_active:
            rospy.loginfo(f"TwistControl: Hinder aktivt -> pausar körning")
        else:
            rospy.loginfo(f"TwistControl: Hinder klart -> återupptar körning")

    def instruction_parse(self):
        """
        Tolkar inkommande instruktionssträng och uppdaterar position och mål.
 
        Förväntat format: "<robot_namn>,x,y,theta,mål_x,mål_y"
        Exempel: "duck4,1.0,0.5,0.0,2.0,1.0"
        """
        while not rospy.is_shutdown():
            if self.instruction != self.prev_instructions:
                self.prev_instructions = self.instruction
                self.current_order_list = self.prev_instructions.split(",") 
                del self.current_order_list[0]      # ta bort namnet på roboten
                self.current_order_list = [float(i) for i in self.current_order_list]
                self._position[0:3] = self.current_order_list[0:3]      # Uppdatera x, y, theta
                self._position[2] = self._position[2]
                self.goal_pose[0:2] = self.current_order_list[3:5]      # Uppdatera målkoordinat
            else:
                break # om inga nya instruktioner på topic, uppdatera inget

    def callback_left(self, data):
        """Tar emot tickvärde från vänster hjulenkodare."""
        self._ticks_left = data.data

    def callback_right(self, data):
        """Tar emot tickvärde från höger hjulenkodare."""
        self._ticks_right = data.data
   
    def _publish_cmd(self, v, omega):      
       """Publicerar ett körkommando till drivtopic."""
       self._publisher.publish(Twist2DStamped(v=v, omega=omega))       

    def normalize_angle(self,angle): 
        """Normaliserar vinkel till intervallet [-pi, pi]."""
        while angle > math.pi:   
            angle -= 2 * math.pi

        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def reset_PID(self):   
        """Nollställer PID-tillståndet. Anropas inför rotation på plats."""
        self._theta_error_integral = 0.0
        self._prev_theta_error     = 0.0
        
    def check_angle_error(self):
        """
        Beräknar normaliserat vinkelfel mot önskad riktning.
        Returns:
            float: Vinkelfel i radianer, normaliserat till [-pi, pi].
        """
        theta_error = self.DESIRED_THETA - self._position[2]
        return self.normalize_angle(theta_error)
    
    def PID_omega(self, theta_error,dt):
        """
        Beräknar PID-styrsignal (omega) för vinkelreglering.
        Args:
            theta_error: Aktuellt vinkelfel [rad].
            dt: Tidssteg [s].
        Returns:
            float: Begränsad vinkelhastighet [rad/s].
        """
        self._derivatan = self.KD_THETA * (theta_error-self._prev_theta_error)/dt
        self._prev_theta_error = theta_error

        self._theta_error_integral += theta_error * dt 
        self._theta_error_integral = max(-5.0, min(5.0, self._theta_error_integral))

        omega = self.KP_THETA * theta_error + self.KI_THETA * self._theta_error_integral + self._derivatan
        omega = max(-OMEGA_MAX, min(OMEGA_MAX, omega)) 
        return omega

    
    def rotation_to_correct(self, rate, dt, prev_ticks_left, prev_ticks_right):
        """
        Roterar roboten på plats tills vinkelfel < Accepted_angle.
        Används när felet är för stort för PID-styrning under framkörning.
        Returns:
            tuple: (prev_ticks_left, prev_ticks_right) uppdaterade enkodervärden.
        """
        rospy.loginfo(
            f"Vinkelfel > {math.degrees(Accepted_angle)}grader -> roterar pa plats "
            f"Nuvarande: {math.degrees(self._position[2]):.1f} grader"
            f"Önskar: {math.degrees(self.DESIRED_THETA):.1f} grader"
        )

        self.reset_PID()
        while not rospy.is_shutdown():
            self.instruction_parse()
            dt, prev_ticks_left, prev_ticks_right = self.update_odometry(
                prev_ticks_left, prev_ticks_right
            )
            theta_error =  self.check_angle_error()
            
            if abs(theta_error) < Accepted_angle:
                rospy.loginfo(f"Vinkle OK!  fel={math.degrees(theta_error):1f} grader")
                break
            
            omega = self.PID_omega(theta_error, dt)
            self._publish_cmd (v=0.00, omega=omega)       

            rospy.loginfo_throttle(
                1, 
                f"Roterar fel {math.degrees(theta_error):.1f} grader omega = {omega:.3f}"
            )
            rate.sleep()
        self.reset_PID()
        return prev_ticks_left, prev_ticks_right

    def straight_forward(self, dt): 
        """
        Kör roboten rakt framåt med PID-styrning mot DESIRED_THETA.
        Används när vinkelfel < Accepted_angle.
        """
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
    
    #==========================
    # Odometri
    #==========================
    def update_odometry(self, prev_tick_left, prev_ticks_right):
        """
        Uppdaterar robotens position med fusionerad enkoder- och gyrodata.
        Enkodrar ger förflyttning, gyro ger rotation.
        Fusioneringen använder ett viktat medelvärde (alpha = 0.85 för gyro).
        Args:
            prev_tick_left:  Tidigare enkodervärde vänster hjul.
            prev_ticks_right: Tidigare enkodervärde höger hjul.
        Returns:
            tuple: (dt, ticks_left, ticks_right)
        """
        current_time = rospy.get_time()
        dt = current_time - self.senast_tid

        # Beräkna förflyttning per hjul [m]
        dNl = self._ticks_left  - prev_tick_left
        dNr = self._ticks_right - prev_ticks_right
        dl = WHEEL_CIRC * (dNl / TICKS_PER_REV)   
        dr = WHEEL_CIRC * (dNr / TICKS_PER_REV)  
        
        d =  (dl + dr) / 2.0                       # Framåtförflyttning [m]       
        dtheta_enc = (dr - dl) / AXIS_LENGTH       # Rotationsuppskattning från enkodrar [rad]
        
        # Fusionera enkoder och gyro för rotationsuppskattning
        alpha = 0.85
        gyro_dtheta = (self.latest_imu_gyro_z-self.gyro_bias)*dt
        fused_dtheta = alpha *gyro_dtheta + (1-alpha) * dtheta_enc 

        # Uppdatera position med midpoint-integration
        midpoint_theta  = self._position[2] + fused_dtheta / 2.0
        self._position[0] += d * math.cos(midpoint_theta)
        self._position[1] += d * math.sin(midpoint_theta)
        self._position[2]  = self.normalize_angle(self._position[2] + fused_dtheta)
        
        self.senast_tid = current_time
        return dt, self._ticks_left, self._ticks_right

    def calculate_desired_direction(self):
        """
        Beräknar önskad körriktning mot målkoordinatet med atan2.
        Hastigheten skalas ner när roboten är nära målet.
        Publicerar desired_theta och current_theta till obstacle_detection_node.
        """        
        dx_g = self.goal_pose[0] - self._position[0]    
        dy_g = self.goal_pose[1] - self._position[1]

        dist_goal = math.sqrt(dx_g**2 + dy_g**2)
        phi_goal = math.atan2(dy_g,dx_g) 
        
        if dist_goal < 0.05: 
            return        # Tillräckligt nära målet - ingen uppdatering behövs
        
        self.DESIRED_THETA = math.atan2(dy_g, dx_g)    
        self.VELOCITY  = min(BASE_SPEED,dist_goal)

        self.desired_theta_pub.publish(Float32(data=self.DESIRED_THETA))


    #=======================
    # Huvudloop
    #=======================
    def run(self):
        """Kör huvudloopen med 25 Hz."""
        rate = rospy.Rate(25)
        
        # Vänta tills enkodrar är initierade
        rospy.loginfo("Väntar på encoder data.")
        while (self._ticks_left is None or self._ticks_right is None) and not rospy.is_shutdown():
            rate.sleep() 

        if rospy.is_shutdown():
            return
         
        # Kalibrera gyro innan körning startar
        self.calibrate_gyro(duration=2.0)
        prev_ticks_left  = self._ticks_left
        prev_ticks_right = self._ticks_right
        self.senast_tid = rospy.get_time()
        rospy.loginfo(f"Encoder OK! start: V= {prev_ticks_left}  H={prev_ticks_right}")
        rospy.loginfo(f"Start -> Mal:{self.goal_pose}")

        while not rospy.is_shutdown():
            self.instruction_parse()
            dt, prev_ticks_left, prev_ticks_right = self.update_odometry(prev_ticks_left, prev_ticks_right)
            
            # Publicera aktuell vinkel för obstacle_detection_nod
            self.current_theta_pub.publish(Float32(data=self._position[2]))
            
            # Pausa om hindernoden är aktiv
            if self.obstacle_active:
                self._publish_cmd(v=0.0, omega=0.0)
                rate.sleep()
                continue

            # Stanna om målet är nått
            if self.Goal_reached():
                rospy.loginfo_throttle(2, "Är i önskade position  :)")
                self._publish_cmd(v=0.0, omega=0.0)
                rate.sleep()
                continue
            
            # Beräkna önskad riktning och styrsignal
            self.calculate_desired_direction()      
            theta_error = self.check_angle_error() 

            if abs(theta_error) > Accepted_angle:
                self._publish_cmd(v=0.0, omega= 0.0)
                prev_ticks_left,prev_ticks_right = self.rotation_to_correct(rate, dt, prev_ticks_left, prev_ticks_right)
            else:
                # Vinkelfel acceptabelt - kör rakt med PID
                self.straight_forward(dt) 

            rate.sleep()

    def on_shutdown(self):
        """Stoppar roboten säkert vid nedstängning."""
        rospy.loginfo("Stoppar Roboten")
        stop = Twist2DStamped(v=0.0, omega=0.0)
        self._publisher.publish(stop)

if __name__ == '__main__':
    node = TwistControlNode(node_name='twist_control_node')
    node.run()
    rospy.spin()
