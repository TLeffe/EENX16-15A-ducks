#!/usr/bin/env python3

import os
import cv2       #  används för att bearbeta kamerabilder
import math
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import Range, CompressedImage
from std_msgs.msg import Bool,Float32
from cv_bridge import CvBridge         # används för att konvertera ROS-bilder till OpenCV


TOF_THRESHOLD   = 0.5    # meter — triggar kameran om föremål är närmare än detta
AVOID_OMEGA     = 2.0    #  hur hårt roboten svänger för att undvika (rad/s )
AVOID_VELOCITY  = 0.2    #  hastighet under undvikande
AVOID_DURATION  = 1.5    # hur länge roboten svänger (sek)
CAMERA_HFOV = 160.0       # grader — kamerans horisontella synfält på Duckiebot
# -------------------------------------------------------
# PI-regulator för styrning 
# -------------------------------------------------------
KP_THETA = 4               # proportionell — hur hårt vi styr mot rätt riktning
KI_THETA =  0.1          # integral — kompenserar konstant drift
OMEGA_MAX = 4.0              # max vridningshastighet (säkerhetsgräns)
KD_THETA= 0.2


class ObstacleDetectionNode(DTROS):

    def __init__(self, node_name):
        super(ObstacleDetectionNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']

        self.twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.tof_topic = f"/{self.vehicle_name}/front_center_tof_driver_node/range"  # ROS-topic för att ta emot data från ToF-sensorn
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"      #   ROS-topic för att ta emot komprimerade bilder från kameran.
        self.obstacle_topic = f"/{self.vehicle_name}/obstacle_detection_node/obstacle_detected"    # ROS-topic för att publicera om ett hinder upptäckts
        self.desired_theta_topic = f"/{self.vehicle_name}/twist_control_node/desired_theta"          # Lyssnar på twist_control för att få vinkeln mot målet

        #----- Tillståndsvariabler
        self.tof_range = float ('inf')     # Senaste avståndet från ToF
        self.camera_active = False         # True när kameran analyserar
        self.obstacle_detected  = False
        
        
        self.last_image = None          # senaste kameranild
        
        self.theta_original = 0.0    # Vinkel mot målet - uppdateras löpande från twist_control
        
        
        # current_theta = robotens faktiska vinkel just nu från twist_controls odometri
        # används av PID så den vet var roboten faktiskt är (inte bara vart den ska)
        self.curren_theta = 0.0
        self.theta_avoid = None      # Vinkel att svänga mot för att passera hindret
        #   PID-sate
        self._integral = 0.0
        self._prev_error = 0.0
        self.obstacle_pub = rospy.Publisher(self.obstacle_topic, Bool, queue_size=1)    #  Publisher som skickar sant/falskt om hinder finns
        self.twist_pub = rospy.Publisher(self.twist_topic, Twist2DStamped, queue_size=1)  # Publisher som skickar hastighetskommandon till roboten
        
        rospy.Subscriber(self.tof_topic, Range, self.callback_tof)      # Lyssnar på ToF-sensorn och kör callback_tof vid nytt meddelande
        rospy.Subscriber(self.camera_topic, CompressedImage, self.callback_camera)   # Lyssnar på kameran och kör callback_camera vid ny bild
        rospy.Subscriber(self.desired_theta_topic, Float32, self.callback_desired_theta)
        rospy.Subscriber(self.current_theta_topic, Float32,self.callback_current_theta)
    

    def callback_desired_theta (self, msg):
        if not self.obstacle_detected:    # Uppdaterar theta_original med önskad vinkel om inget hinder detekteras
            self.theta_original= msg.data

    def callback_current_theta(self, msg):   # Tar emot robotens aktuella vinkel för PID-reglering
        self.curren_theta = msg.data
        
    def callback_tof(self, msg):   #  Körs varje gång ToF-sensorn skickar ett nytt avstånd.

        self.tof_range = msg.range      # Sparar senaste avståndet
        if msg.range < TOF_THRESHOLD:         # Kolla om avståndet är mindre än tröskelvärdet
            if not self.camera_active:           # Om kameran inte redan är aktiv
                rospy.loginfo(f"TOF: Formål pa {msg.range:.2f}m --> aktivera kamera")
                self.camera_active = True
        else:
            if self.camera_active:             # Om avståndet är större än tröskeln
                rospy.loginfo(f"ToF: Vägen klar ({msg.range:.2f}m) — kamera inaktiv")
                self.camera_active = False     # Inget hinder nära --> stäng av kamera-analys
                self.obstacle_detected = False

    def callback_camera(self, msg):     #   Körs varje gång kameran skickar en ny bild.

        if not self.camera_active:      # Hoppa över om kameran inte är aktiv
            return
        if self.obstacle_detected:   # undvikande pågår redan — vänta
            return
        
        try: 

            # Gör om den komprimerade bilden till OpenCV-format
            np_arr = np.frombuffer(msg.data, np.uint8)   # np.frombuffer = konverterar dess till en numpy-arry  (np.unit8= varje värde tolkas som ett tal 0-225)

            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)   # cv2.imdecode() tar numpy-arry och dekoder den till en riktig bild / cv2. IMREAD_COLOR = säger att bilden ska läsas som en färgbild
            if image is None:
                return  
          #  self.last_image = image
            
            obstacle_found, direction, edge_angle = self.analys_image(image)         # Analyserar bilden


            # Om hinder hittas 
            if obstacle_found:
                rospy.loginfo("Kamera: Foremål bekräftat — undviker!")
                self.obstacle_detected = True
                self.theta_avoid = edge_angle

    
                self.obstacle_pub.publish(Bool(data=True))    # signalera att hinder finns
                self.avoid_obstacle(direction)
        except Exception as e:    # Om något går fel--> krascha inte, skriv ut ett felmeddelande istället
            rospy.logwarn(f"Kamerafel: {e}")


    def analys_image (self, image):   # analysera  komerabild för att bekräfta föremål. 
        
        if image is None: 
            return False, 0, 0.0
      
        height, width = image.shape[:2]    # Hämtar bildens höjd och bredd 
        
        #  Kolla om det finns en stor kontur i mitten av bilden
        center_x = width // 2
        center_y = height // 2 
        # vi förenklar bilden (mindre data) som gör bildanalys enklare
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # Gör bilden gråskalig/ COLOR_BGR2GRAY konverterar bilden från BEG TO RGB / cv2.cvtColor --> ändra färgeformat på en bild 
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Suddar lite för att minska brus/(5, 5)  lagom blur vi har 0 då vi låter opencv välja bästa styrkan auto
        edges = cv2.Canny(blurred, 50, 150)        # Hittar kanter i bilden,  <50 → ignorera, 50–150 → kanske viktigt, 150 → definitivt viktig / (50-150) --> ta bara tydliga kanter, men tillåt lite svagare kant
        contours, hierarchy  = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # hittar kanter i bilden
        '''cv2.findContours()--> tar en edge-bild, hittar alla objekt, returnerar deras former
        cv2.RETR_EXTERNAL --> tar bara yttersta konturer/ cv2.CHAIN_APPROX_SIMPLE--> spara bara vikiga punker
        hierarchy -->  vi ignorerar den andra just nu / contours --> list av objekt '''   
       
        # Hitta den STÖRSTA konturen i mitten av bilden
        best_contour = None   
        best_area = 0

        for contour in contours: 
            area = cv2.contourArea(contour)   # Storlek på objektet

            if area < 1000: # Ignorera små konturer (brus)
                continue

            # Hämtar en rektangel runt konturen
            x , y,w,h = cv2.boundingRect(contour)    # Lätt att hitta mitten av objektet
            contour_center_x = x + w // 2      # konturens mittpunkt x
            contour_center_y = y + h // 2      # konturens mittpunkt y
            
            # Om konturen är i mitten +,-30% av bilden --> föremål bekräftat
            # Vi bryr oss bara om objekt i mitten (t.ex. hinder rakt fram)
            within_x = abs(contour_center_x - center_x) < width  * 0.3
            within_y = abs(contour_center_y - center_y) < height * 0.3            

            if within_x and within_y and area > best_area:
                best_area    = area
                best_contour = contour            

        if best_contour is None: 
            return False, 0, 0.0
        
         # Beräkna hur mycket utrymme som finns på var sida om hindret
        x , y,w,h = cv2.boundingRect(best_contour)
        
        left_space = x                  # plats från föremålet till vänster kant
        right_space = width -(x+w)       # plats från föremålet till höger kant
        rospy.loginfo(
            f"Foremål hittad: area={best_area:.0f}  "
            f"plats vänster={left_space}  höger={right_space}"
        )
        # Sväng mot den sida som har mest plats
        if left_space > right_space:
            edge_pixel = x    # hindrets vänstra kant
            direction = +1
            rospy.loginfo("Mest plats at VÄNSTER — svänger vänster")
        else:
            edge_pixel = x + w   # hindrets högra kant
            direction= -1
            rospy.loginfo("Mest plats at HÖGER — svänger höger")
        
        pixel_offset = edge_pixel - (width/ 2.0)   # Beräknar avstånd från bildens mitt till hinderkanten i pixlar
        angle_deg = (pixel_offset/(width/2.0)) * (CAMERA_HFOV /2.0)   # Omvandlar pixelavstånd till en vinkel i grader
        edge_angle = self.curren_theta + math.radians(angle_deg)     # Beräknar den absolut vinkel i världen

        return True, direction, edge_angle
    

    def avoid_obstacle(self, direction):   # undvik förmål genom att sbänga 

       # side = "vänster" if direction > 0 else "höger"
        # Gör riktningen till text för loggning

       # rospy.loginfo(f"Undviker foremål — svänger {side}")
        
        # Spara starttiden för undanmanövern
        #start_time = rospy.Time.now()
        rate = rospy.Rate(20)
        self.reset_pid()

        while not rospy.is_shutdown():
            if not self.obstacle_in_center():
                rospy.loginfo(f"FAS 1 klar------>> hindret lämnat mittzon :)")
                break
            omega = self.pid_steer(self,self.theta_avoid)
            self.twist_pub.publish(Twist2DStamped(v=AVOID_VELOCITY, omega=omega))
            rate.sleep()

        self.twist_pub.publish(Twist2DStamped(v=0.0, omega=0.0))   # här stannar vi 0.3 sec efter att vi har passerat hinder
        rospy.sleep(0.3)
        self.reset_pid() 

        rospy.loginfo(f"FAS 2 roterar tillbaka till theta_original={math.degrees(self.theta_original):.1f}")
        while not rospy.is_shutdown():
            omega = self.pid_steer(self,self.theta_avoid)
            angle_error = self.angle_error_to(self.theta_original)
            self.twist_pub.publish(Twist2DStamped(v=AVOID_VELOCITY, omega=omega))
            if abs(angle_error) < math.radians(5):  # om den är mindre än 5 grader
                rospy.loginfo(f"Fas 2 klar----->> tillbaka till ursprungsvinkel")
                break
            rate.sleep()
        
        self.twist_pub.publish(Twist2DStamped(v=0.0, omega=0.0))   # Återställ och signalera till twist_control
        self.obstacle_detected = False
        self.camera_active = False
        self.theta_avoid = None
        self.obstacle_pub.publish(Bool(data=False))
        rospy.loginfo(f"Undvikande är klar och twist_control återsupptar")
        



            # Räkna ut hur lång tid som gått sedan undvikandet började
      #      elapsed = (rospy.Time.now() - start_time).to_sec()

       #     if elapsed < AVOID_DURATION:
                # Om vi fortfarande är inom undvikandetiden
                
        #        self.twist_pub.publish(Twist2DStamped(v=AVOID_VELOCITY, omega=direction * AVOID_OMEGA))
                # omega = rotation, vänster eller höger beroende på direction

         #       rospy.loginfo_throttle(
          #          1, f"Undviker... {elapsed:.1f}/{AVOID_DURATION}s"
           #     )
                # Skriv logg högst var 1 sekund så att loggen inte spammas

         #   else:   # Stäng av kameran igen

          #      rospy.loginfo("Undvikande klart — förtsätter")
                # Logga att roboten är klar med att svänga undan

           #     self.obstacle_pub.publish(Bool(data=False))    # Publicera att hinder inte längre aktivt undviks
            #    self.obstacle_detected = False   # Nollställ hinderstatus
             #   self.camera_active = False    # Stäng av kameran igen

              #  break
           # rate.sleep()


    def obstacle_in_center(self):   # kolla om hinder ff syns i kamera mittzon
        try: 
            msg    = rospy.wait_for_message(self.camera_topic, CompressedImage, timeout=0.5)
            np_arr = np.frombuffer(msg.data, np.uint8)
            image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            found, _, _ = self.analys_image(image)
            return found
        except rospy.ROSException:
            return True    # timeout -> anta att hindret fortfarande finns
        
    def pid_steer(self, target_theta):
        dt = 1.0/20.0
        error = self.angle_error_to (target_theta)

        derivatan = KD_THETA * (error - self._prev_error)/dt
        self._prev_error = error

        self._integral += error * dt   # uppdatera integralen (I-delen)
        self._integral = max(-8.0, min(8.0, self._integral))

        omega = KP_THETA * error + KI_THETA * self._integral + derivatan
        omega = max(-OMEGA_MAX, min(OMEGA_MAX, omega))     # roboten ska inte vrider sig för snabbt
        
        return omega
    
    def angle_error_to(self, target):      # Normalisera felet till [-pi, pi], vi anväder nuvarnde theta 
        error = target -self.curren_theta
        while error > math.pi:   
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi
        return error   
    
    def reset_pid(self):
        self._integral = 0.0
        self._prev_error = 0.0
        
    def run (self): 
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            rospy.loginfo_throttle(
                3.0,
                f"ToF: {self.tof_range:.2f}m  "
                f"Kamera: {'aktiv' if self.camera_active else 'inaktiv'}  "
                f"Hinder: {'JA' if self.obstacle_detected else 'nej'}"
                f"theta_orig: {math.degrees(self.theta_original):.1f}  "
                f"theta_curr: {math.degrees(self.curren_theta):.1f} "
            )
            rate.sleep()
 
    def on_shutdown(self):
        rospy.loginfo("Stoppar ObstacleDetectionNode")
        try:
            for _ in range(5):
                self.twist_pub.publish(Twist2DStamped(v=0.0, omega=0.0))
                rospy.sleep(0.1)
        except:
            pass

 
if __name__ == '__main__':
    node = ObstacleDetectionNode(node_name='obstacle_detection_node')
    node.run()
    rospy.spin()  