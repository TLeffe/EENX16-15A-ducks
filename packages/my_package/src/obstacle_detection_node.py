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


TOF_THRESHOLD = 0.45           # meter - triggar kameran om föremål är närmare än detta
CAMERA_HFOV = 160.0             # grader - kamerans horisontella synfält på Duckiebot
AVOID_VELOCITY = 0.15            #  hastighet under undvikande
REVERSE_DURATION = 0.15 / AVOID_VELOCITY     # Tid för att backa 15 cm

MAX_AVOID_ANGLE = math.radians(30)       # Max vinkel för undvikande i radianer

# -------------------------------------------------------
# PI-regulator för styrning 
# -------------------------------------------------------
KP_THETA = 2               # proportionell — hur hårt vi styr mot rätt riktning
KI_THETA =  0.05          # integral — kompenserar konstant drift
KD_THETA= 0.08
OMEGA_MAX = 1.3              # max vridningshastighet (säkerhetsgräns)
MIN_CONTOUR_AREA = 1000      # Minsta area för att räkna som objekt
ANGLE_THRESHOLD  = math.radians(5)   # Vinkelgräns för att anses vara framme
SCAN_FRAMES = 4                      # Antal bilder att samla in vid scanning
SCAN_TIMEOUT = 2.0               # Max tid för scanning i sekunder
MIN_AVOID_CYCLES = 12             # Minsta antal cykler i undvikande
WALL_THRESHOLD= 100   # pixlar- både sidor under detta = vägg

IDLE = "IDLE"         # Ingen aktivitet, lyssnar på ToF
SCANNING = "SCANNING"
AVOIDING = "AVOIDING"   # Svänger förbi ett hinder
ROTATING = "ROTATING"
REVERSING = "REVERSING"
RETURNING = "RETURNING"  # Roterar tillbaka till originalvinkeln



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
        self.current_theta_topic = f"/{self.vehicle_name}/twist_control_node/current_theta"

        #----- Tillståndsvariabler
        self.tof_range = float ('inf')     # Senaste avståndet från ToF
        self.latest_image = None
        self.camera_active = False         # True när kameran analyserar
        
        self.theta_original = 0.0         # Vinkel mot målet - uppdateras löpande från twist_control
        self.current_theta = 0.0
        self.theta_avoid = 0.0            # Vinkel att svänga mot för att passera hindret
        
        self.state = IDLE            # State machine — börjar i IDLE
        self.scan_results= []        # Lista med undvikande vinklar
        self.scan_start= 0.0
        self.avoid_cycles   = 0      # Räknare för undvikande cykler
        self.reverse_start  = 0.0    # Starttid för backning
        self._integral = 0.0              #   PID-sate
        self._prev_error = 0.0

        self.obstacle_pub = rospy.Publisher(self.obstacle_topic, Bool, queue_size=1)    #  Publisher som skickar sant/falskt om hinder finns
        self.twist_pub = rospy.Publisher(self.twist_topic, Twist2DStamped, queue_size=1)  # Publisher som skickar hastighetskommandon till roboten
        
        rospy.Subscriber(self.tof_topic, Range, self.callback_tof)      # Lyssnar på ToF-sensorn och kör callback_tof vid nytt meddelande
        rospy.Subscriber(self.camera_topic, CompressedImage, self.callback_camera)   # Lyssnar på kameran och kör callback_camera vid ny bild
        rospy.Subscriber(self.desired_theta_topic, Float32, self.callback_desired_theta)
        rospy.Subscriber(self.current_theta_topic, Float32,self.callback_current_theta)

        rospy.loginfo(f"ObstacleDetectionNode startad")

    def callback_desired_theta (self, msg):
        if self.state == IDLE:               # Behåll originalvinkeln fryst under undvikande
            self.theta_original= msg.data     # Under AVOIDING/RETURNING ska den INTE skrivas över — vi vill minnas vart vi skulle

    def callback_current_theta(self, msg):   # Tar emot robotens aktuella vinkel för PID-reglering
        self.current_theta = msg.data
        
    def callback_tof(self, msg):   #  Körs varje gång ToF-sensorn skickar ett nytt avstånd.

        self.tof_range = msg.range      # Sparar senaste avståndet
        if msg.range < TOF_THRESHOLD:         # Kolla om avståndet är mindre än tröskelvärdet
            if not self.camera_active:          # kolla om kameran inte redan är aktiv
                self.camera_active = True
                rospy.loginfo(f"TOF: Formål på {msg.range:.2f}m ----> aktivera kamera")
        else:
            if self.state == IDLE:             # Om avståndet är större än tröskeln
                rospy.loginfo(f"ToF: Vägen klar ({msg.range:.2f}m) ------> kamera inaktiv")
                self.camera_active = False     # Inget hinder nära --> stäng av kamera-analys

    def callback_camera(self, msg):     #   Körs varje gång kameran skickar en ny bild.

        if not self.camera_active:      # Hoppa över om kameran inte är aktiv
            return
    
        try: 
            # Gör om den komprimerade bilden till OpenCV-format
            np_arr = np.frombuffer(msg.data, np.uint8)   # np.frombuffer = konverterar dess till en numpy-arry  (np.unit8= varje värde tolkas som ett tal 0-225)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)   # cv2.imdecode() tar numpy-arry och dekoder den till en riktig bild / cv2. IMREAD_COLOR = säger att bilden ska läsas som en färgbild
            if image is not None:
                self.latest_image = image
        except Exception as e:    # Om något går fel--> krascha inte, skriv ut ett felmeddelande istället
            rospy.logwarn(f"Kamerafel --------> INTE BRA : {e}")

    def stop (self):     # Funktion för att stoppa roboten
        self.twist_pub.publish(Twist2DStamped(v=0.0, omega=0.0))

    def normalize_angle(self, angle):      # Normaliserar vinkel till intervallet [-pi, pi]
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle 
    
    def angle_error_to(self, target_theta):     # Beräknar vinkelfel mot mål
        return self.normalize_angle(target_theta - self.current_theta)    # Returnerar negativt = vrida höger, positivt = vrida vänster

 #   def blend_angle(self, old_angle, new_angle, alpha):  
   #     """
    #    Filtrera vinklar korrekt via vinkelfel,
     #   inte via vanlig linjär interpolation på råa vinklar.
      #  """
     #   error = self.normalize_angle(new_angle - old_angle)    # Kortaste skillnaden
  #      return self.normalize_angle(old_angle + alpha * error)   # Flytta lite mot nya
    
    def reset_pid(self): 
        self._integral = 0.0
        self._prev_error = 0.0

    def pid_steer(self, target_theta):
        dt = 1.0 / 20.0
        error = self.angle_error_to(target_theta)

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        self._integral += error * dt
        self._integral = max(-3.0, min(3.0, self._integral))

        omega = KP_THETA * error + KI_THETA * self._integral + KD_THETA * derivative
        omega = max(-OMEGA_MAX, min(OMEGA_MAX, omega))
        return omega

    def rotate_to(self, target_theta):        # Roterar mot en given vinkel
        if abs(self.angle_error_to(target_theta)) < ANGLE_THRESHOLD:    # Om nära nog
            self.stop()
            return True
        self.twist_pub.publish(Twist2DStamped(v=0.0, omega=self.pid_steer(target_theta)))   # Roterar på plats
        return False
    
    def analys_image (self, image):   # analysera  komerabild för att bekräfta föremål. 
        
        if image is None: 
            return False, 0.0  , False
      
        height, width = image.shape[:2]    # Hämtar bildens höjd och bredd 
        
        #  Kolla om det finns en stor kontur i mitten av bilden
        center_x = width // 2
        center_y = height // 2 
        # vi förenklar bilden (mindre data) som gör bildanalys enklare
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # Gör bilden gråskalig/ COLOR_BGR2GRAY konverterar bilden från BEG TO GRAY / cv2.cvtColor --> ändra färgeformat på en bild 
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Suddar lite för att minska brus/(5, 5)  lagom blur vi har 0 då vi låter opencv välja bästa styrkan auto
        edges = cv2.Canny(blurred, 50, 150)        # Hittar kanter i bilden,  <50 → ignorera, 50–150 → kanske viktigt, 150 → definitivt viktig / (50-150) --> ta bara tydliga kanter, men tillåt lite svagare kant
        
        contours, _  = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # hittar kanter i bilden
        '''cv2.findContours()--> tar en edge-bild, hittar alla objekt, returnerar deras former
        cv2.RETR_EXTERNAL --> tar bara yttersta konturer/ cv2.CHAIN_APPROX_SIMPLE--> spara bara vikiga punker'''   
       
        best_contour = None       
        best_area = 0

        for contour in contours:   
            area = cv2.contourArea(contour)   # Storlek på objektet
            if area < MIN_CONTOUR_AREA: # Ignorera små konturer (brus)
                continue

            # Hämtar en rektangel runt konturen
            x , y,w,h = cv2.boundingRect(contour)    # Lätt att hitta mitten av objektet
            contour_center_x = x + w // 2      # konturens mittpunkt x
            contour_center_y = y + h // 2      # konturens mittpunkt y
            
            # Om konturen är i mitten +,-30% av bilden --> föremål bekräftat
            # Vi bryr oss bara om objekt i mitten (t.ex. hinder rakt fram)
            within_x = abs(contour_center_x - center_x) < width  * 0.3
            within_y = abs(contour_center_y - center_y) < height * 0.3            

            if within_x and within_y and area > best_area:    # Spara om denna kontur är störst hittills och är i mittzon
                best_area    = area
                best_contour = contour            

        if best_contour is None:   # Inga hinder hittades i mittzon

            return False, 0.0 , False
        
         # Beräkna hur mycket utrymme som finns på var sida om hindret
        x , y,w,h = cv2.boundingRect(best_contour)
        
        left_space = x                  # plats från föremålet till vänster kant
        right_space = width -(x+w)       # plats från föremålet till höger kant
        rospy.loginfo(
            f"Foremål hittad: area={best_area:.0f}  "
            f"plats vänster={left_space}  höger={right_space}"
        )
        # Sväng mot den sida som har mest plats
        if left_space < WALL_THRESHOLD and  right_space < WALL_THRESHOLD:
            rospy.loginfo_throttle(1.0, f"VÄGG  |  vänster={left_space} | höger={right_space} --> backar")
            return True, 0.0, True
        if left_space > right_space:
            edge_pixel = x
            side_text = "VÄNSTAR"
            rospy.loginfo("Mest plats på VÄNSTAR --->>> svänger vänstar")
        else:
            edge_pixel = x + w   # hindrets högra kant
            side_text = "HÖGER"
            rospy.loginfo("Mest plats på HÖGER --->>> svänger höger")
        
        pixel_offset = (edge_pixel- width/2.0) /(width/2.0) * (CAMERA_HFOV /2)  # Pixel till vinkel
        angle_error = max(-MAX_AVOID_ANGLE, min(MAX_AVOID_ANGLE, math.radians(pixel_offset)))    # Begränsa vinkel
        theta_avoid = self.normalize_angle(self.current_theta + angle_error)       # Beräkna undvikande riktning
        
        rospy.loginfo_throttle(
            1.0,
            f"Hinder | sida={side_text} | vänster={left_space}px | "
            f"höger={right_space}px | theta_avoid={math.degrees(theta_avoid):.1f}"
        )


        return True, theta_avoid, False
    
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()    # Hämtar nuvarande ROS-tid i sekunder
            if self.state ==IDLE:   # IDLE--->> väntar på hinder
                if self.camera_active and self.latest_image is not None:  # Om kamera är aktiv och bild finns
                    self.obstacle_pub.publish(Bool(data=True))    # Publicera att hinder finns
                    self.stop()
                    self.scan_results= []     # Töm gamla scanresultat
                    self.scan_start = now     # Spara scan-starttid
                    self.state = SCANNING
                    rospy.loginfo(f"IDLE--->SCANNING | theta_orig={math.degrees(self.theta_original):.1f}")
            
            elif self.state== SCANNING:
                self.stop()
                if self.latest_image is not None:  # Om bild finns
                    found, theta, is_wall = self.analys_image(self.latest_image)
                    if found and is_wall:     # Om vägg upptäcks
                        self.reverse_start = now      # Spara starttid för backning
                        self.reset_pid()     # Nollställ PID
                        self.state= REVERSING
                        rospy.loginfo("SCANNING --> REVERSING (vägg)")
                        rate.sleep(); continue     # Vänta och hoppa till nästa loop
                    if found:
                        self.scan_results.append(theta)     # Spara undvikande vinkel
                
                ready = len(self.scan_results) >= SCAN_FRAMES     # Kolla om tillräckligt många bilder samlats
                timeout = (now -self.scan_start) >= SCAN_TIMEOUT   # Kolla om tiden tagit slut

                if ready or (timeout and self.scan_results):       # Om redo eller timeout med resultat
                    self.theta_avoid = float(np.median(self.scan_results))      # Ta median av vinklarna
                    self.reset_pid()
                    self.state= ROTATING
                    rospy.loginfo(f"SCANNING --> ROTATING | theta_avoid={math.degrees(self.theta_avoid):.1f}grader")
                elif timeout:      # Om timeout utan resultat
                    self.obstacle_pub.publish(Bool(data=False))   # Publicera inget hinder
                    self.camera_active = False
                    self.state= IDLE
                    rospy.loginfo("SCANNING -->> IDLE (timeout, inget hinder)")
            
            elif self.state == ROTATING:        # Om tillstånd är ROTATING
                if self.rotate_to(self.theta_avoid):      # Roterar mot undvikande riktning
                    self.avoid_cycles= 0          # Nollställ cykelräknare
                    self.reset_pid()
                    self.state =AVOIDING
                    rospy.loginfo("ROTATING --->> AVOIDING")
            elif self.state == AVOIDING:
                if self.latest_image is None:
                    rate.sleep(); continue
                
                self.avoid_cycles +=1     # Öka räknaren
                found,_, is_wall = self.analys_image(self.latest_image)
                if is_wall:      # Om vägg upptäcks
                    self.stop()
                    self.reverse_start = now
                    self.reset_pid()
                    self.state= REVERSING
                    rospy.loginfo("AVOIDING -->>> REVERSING (vägg)")
                elif found or self.avoid_cycles < MIN_AVOID_CYCLES:  # Hinder syns ELLER inte kört tillräckligt - fortsätt framåt
                    self.twist_pub.publish(Twist2DStamped(v=AVOID_VELOCITY, omega=self.pid_steer(self.theta_avoid)))     # Kör framåt och styr
                    rospy.loginfo_throttle(2.0, f"AVOIDING | cykler={self.avoid_cycles} | current={math.degrees(self.current_theta):.1f}")
                else:     # Om hindret är förbi
                    self.stop()
                    rospy.sleep(0.25)   # Kort paus
                    self.reset_pid()
                    self.state = RETURNING
                    rospy.loginfo("AVOIDING -->> RETURNING")
            
            elif self.state == REVERSING:
                if (now - self.reverse_start) < REVERSE_DURATION:    # Om backtiden inte är klar
                    self.twist_pub.publish(Twist2DStamped(v=-AVOID_VELOCITY, omega=0.0))  # Backa rakt bakåt
                else:   # När backning är klar
                    self.stop()
                    rospy.sleep(0.25)
                    self.reset_pid()
                    self.state=RETURNING
                    rospy.loginfo("REVERSING -->>>> RETURNING")

            elif self.state == RETURNING:
                if self.tof_range < TOF_THRESHOLD and self.latest_image is not None:  # Om nytt hinder är nära
                    found,_,_ =self.analys_image(self.latest_image)
                    if found:   # Om nytt hinder hittas
                        self.scan_results=[]   # Töm gamla resultat
                        self.scan_start = now
                        self.stop()
                        self.state = SCANNING
                        rospy.loginfo("RETURNING -->> SCANNING (nytt hinder)")
                        rate.sleep(); continue
                if self.rotate_to(self.theta_original):    # Roterar tillbaka till ursprunglig riktning
                    self.obstacle_pub.publish(Bool(data=False))
                    self.camera_active = False
                    self.state = IDLE
                    rospy.loginfo(f"RETURNING -->> IDLE | theta_orig={math.degrees(self.theta_original):.1f}")

            rospy.loginfo_throttle(3.0, f"[{self.state}] ToF={self.tof_range:.2f}m | orig={math.degrees(self.theta_original):.1f} curr={math.degrees(self.current_theta):.1f}")
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