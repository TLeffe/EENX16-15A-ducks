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

#------Avståndsgränser
TOF_THRESHOLD = 0.45           # meter - triggar kameran om föremål är närmare än detta
TOF_WALL_DIST = 0.17   # meter - under detta = vägg, backa direkt utan kamera
TOF_CLEAR_DIST  = 0.50   # meter - används i ToF-fallback scan
FALLBACK_DRIVE  = 0.35   # meter - hur långt vi kör i fallback-läge

#-----Hastigheter-------
AVOID_VELOCITY = 0.15            #  hastighet under undvikande
REVERSE_SPEED = 0.15           #  # m/s bakåt
REVERSE_DURATION = REVERSE_SPEED / AVOID_VELOCITY     # Tid för att backa 20 cm
REVERSE_DISTANCE = 0.20 # m 

#---- camera------

MAX_AVOID_ANGLE = math.radians(30)       # Max vinkel för undvikande i radianer
CAMERA_HFOV = 160.0             # grader - kamerans horisontella synfält på Duckiebot
MIN_CONTOUR_AREA = 1000      # Minsta area för att räkna som objekt
WALL_THRESHOLD = 80
SCAN_FRAMES = 4                      # Antal bilder att samla in vid scanning
SCAN_TIMEOUT = 2.0               # Max tid för scanning i sekunder

# ----- PID--
KP_THETA = 2               # proportionell — hur hårt vi styr mot rätt riktning
KI_THETA =  0.05          # integral — kompenserar konstant drift
KD_THETA= 0.08
OMEGA_MAX = 1.3              # max vridningshastighet (säkerhetsgräns)

# ---- vinkeltoleranser 
ANGLE_THRESHOLD  = math.radians(5)   # Vinkelgräns för att anses vara framme
FALLBACK_SCAN_STEP   = math.radians(8)   # rad - stegstorlek vid ToF-fallback scan
FALLBACK_MAX_ROTATE  = math.radians(160) # rad - max rotation under fallback scan

#---- minimiköravstånd 
MIN_AVOID_DISTANCE = 0.25   # meter mi-nsta köravstånd

#---- Tillstånd------- 
IDLE = "IDLE"         # Ingen aktivitet, lyssnar på ToF
SCANNING = "SCANNING"
AVOIDING = "AVOIDING"   # Svänger förbi ett hinder
ROTATING = "ROTATING"
REVERSING = "REVERSING"
RETURNING = "RETURNING"  # Roterar tillbaka till originalvinkeln
SCAN_FALLBACK = "SCAN_FALLBACK"   # ToF-rotering när kameran inte hittar hinder
FB_DRIVING    = "FB_DRIVING"      # Kör 35 cm i fallback-riktning



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

        # Vinklar
        self.theta_original          = 0.0  # vinkeln mot målet - fryses i IDLE
        self.current_theta           = 0.0  # robotens faktiska vinkel (odometri)
        self.theta_avoid             = 0.0  # bestäms i SCANNING, låst sedan
        self.theta_at_rotation_start = 0.0  # för att spåra hur mycket vi roterat
        self.total_rotation_done     = 0.0  # hur många radianer vi roterat totalt

        # Köravstånd (Pythagoras)
        self.avoid_distance  = MIN_AVOID_DISTANCE
        self.distance_driven = 0.0

        self.state = IDLE            # State machine — börjar i IDLE

        self.scan_results= []        # Lista med undvikande vinklar
        self.scan_start= 0.0
        self.reverse_start  = 0.0    # Starttid för backning

        self.fallback_scan_dir = -1.0   # -1 = höger, +1 = vänster
        self.fallback_scan_start_theta = 0.0  # vinkel när fallback-scan startade
        self.fallback_drive_start      = 0.0  # tid när fallback-körning startade
        self.fallback_drive_distance   = 0.0

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
        if msg.range < TOF_WALL_DIST and self.state in (IDLE, SCANNING, RETURNING):         # Kolla om avståndet är mindre än 0.17
            rospy.loginfo(f"ToF VÄGG: {msg.range:.2f}m < {TOF_WALL_DIST}m --->>> backar direkt")
            self.obstacle_pub.publish(Bool(data=True))
            self.stop()
#            self.reverse_start= rospy.Time.now().to_sec()
 #           self.reset_pid()
            self.start_reversing()
#            self.state=REVERSING
            return
        if msg.range < TOF_THRESHOLD:
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
        while angle >  math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle 
    
    def angle_error_to(self, target):     # Beräknar vinkelfel mot mål
        return self.normalize_angle(target - self.current_theta)    # Returnerar negativt = vrida höger, positivt = vrida vänster
    
    def reset_pid(self): 
        self._integral = 0.0
        self._prev_error = 0.0

    def pid_steer(self, target, dt = 0.1):
        error = self.angle_error_to(target)
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        self._integral  = max(-3.0, min(3.0, self._integral + error * dt))
        omega = KP_THETA * error + KI_THETA * self._integral + KD_THETA * derivative
        omega = max(-OMEGA_MAX, min(OMEGA_MAX, omega))
        return omega

    def rotate_to(self, target):        # Roterar mot en given vinkel
        if abs(self.angle_error_to(target)) < ANGLE_THRESHOLD:    # Om nära nog
            self.stop()
            return True
        self.twist_pub.publish(Twist2DStamped(v=0.0, omega=self.pid_steer(target)))   # Roterar på plats
        return False
    
    def compute_avoid_distance(self, tof_dist, edge_angle_deg):    # använder av Pythagoras
        a = max(tof_dist, 0.05)  # undvik division med noll
        b = a * math.tan(math.radians(min(abs(edge_angle_deg), 89.0)))   # abs() gör vinkeln positiv, min() begränsar till 89°
        c = math.sqrt(a**2 + b**2) + 0.10   #  10 cm exgtra 
        result = max(MIN_AVOID_DISTANCE, c)
        rospy.loginfo(
            f"Pythagoras: a={a:.2f}m | b={b:.2f}m | "
            f"c={c:.2f}m --->> kör {result:.2f}m"
        )
        return result
    def start_reversing (self):  # Sätt tillstånd REVERSING och spara starttid.
        self.reverse_start = rospy.time.now().to_sec()
        self.reset_pid()
        self.state= REVERSING

    def analys_image (self, image):   # analysera  komerabild för att bekräfta föremål. 
        if image is None: 
            return False, 0.0  , False , 0.0
      
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

            return False, 0.0 , False , 0.0
        
         # Beräkna hur mycket utrymme som finns på var sida om hindret
        x , y,w,h = cv2.boundingRect(best_contour)
        left_space = x                  # plats från föremålet till vänster kant
        right_space = width -(x+w)       # plats från föremålet till höger kant
        rospy.loginfo(
            f"Foremål hittad: area={best_area:.0f}  "
            f"plats vänster={left_space}  höger={right_space}"
        )
        # Sväng mot den sida som har mest plats
        if left_space < WALL_THRESHOLD and right_space < WALL_THRESHOLD:
            rospy.loginfo_throttle(1.0, f"VÄGG från kamera bild  |  vänster={left_space} | höger={right_space} --> backar")
            return True, 0.0, True, 0.0
        
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
        edge_angle_deg = math.degrees(abs(angle_error))  #Konverterar vinkeln från radianer till grader och gör den positiv.


        rospy.loginfo_throttle(
            1.0,
            f"Hinder | sida={side_text} | vänster={left_space}px | "
            f"höger={right_space}px | theta_avoid={math.degrees(theta_avoid):.1f}| "
            f"kant-vinkel={edge_angle_deg:.1f}grader"
        )


        return True, theta_avoid, False, edge_angle_deg
    
    def run(self):
        rate = rospy.Rate(10)
        dt_avoid = 1.0 / 10.0

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
                    found, theta, is_wall, edge_deg = self.analys_image(self.latest_image)
                    if found and is_wall:     # Om vägg upptäcks
                        rospy.loginfo("SCANNING --> REVERSING (vägg via kamera)")
                        self.start_reversing()
                       # self.reverse_start = now      # Spara starttid för backning
                       # self.reset_pid()     # Nollställ PID
                        #self.state= REVERSING
                        rate.sleep(); continue     # Vänta och hoppa till nästa loop
                    if found:
                        self.scan_results.append((theta, edge_deg, self.tof_range))     # Spara undvikande vinkel
                
                ready = len(self.scan_results) >= SCAN_FRAMES     # Kolla om tillräckligt många bilder samlats
                timeout = (now -self.scan_start) >= SCAN_TIMEOUT   # Kolla om tiden tagit slut

                if ready or (timeout and self.scan_results):       # Om redo eller timeout med resultat
                    thetas    = [r[0] for r in self.scan_results]  # Plockar ut ALLA första värden (theta-vinklar)
                    edge_degs = [r[1] for r in self.scan_results]  #  Plockar ut ALLA andra värden (kantvinklar i grader)
                    tof_vals  = [r[2] for r in self.scan_results]  # Plockar ut ALLA tredje värden (ToF-avstånd)

                    self.theta_avoid = float(np.median(thetas))      # Ta median av vinklarna
                    median_egde_deg = float(np.median(edge_degs)) 
                    median_tof = float(np.median(tof_vals)) 

                    self.avoid_distance = self.compute_avoid_distance(median_tof, median_egde_deg)  # # Beräkna körsträcka och spara startvinkel
                    self.theta_at_rotation_start = self.current_theta
                    self.distance_driven = 0.0
                    self.reset_pid()
                    self.state= ROTATING
                    rospy.loginfo(f"SCANNING --> ROTATING |"
                                  f"theta_avoid={math.degrees(self.theta_avoid):.1f} | "
                                  f"köravstånd={self.avoid_distance:.2f}m"
                                  )
                elif timeout:      # Om timeout utan resultat
                    rospy.loginfo("SCANNING -->>> SCAN_FALLBACK (kamera hittade inget, använder ToF)")
                    self.fallback_scan_start_theta = self.current_theta
                    self.fallback_scan_dir = -1.0  # Börja med att rotera åt höger
                    self.reset_pid()
                    self.state=SCAN_FALLBACK
                  #  self.obstacle_pub.publish(Bool(data=False))   # Publicera inget hinder
                   # self.camera_active = False
                   # self.state= IDLE
                   # rospy.loginfo("SCANNING -->> IDLE (timeout, inget hinder)")
            elif self.state ==SCAN_FALLBACK:
                rotated_so_far = abs(self.normalize_angle(self.current_theta -self.fallback_scan_start_theta))

                if self.tof_range > TOF_CLEAR_DIST:
                    self.theta_avoid = self.current_theta
                    self.fallback_drive_start = now
                    self.fallback_drive_distance = 0.0
                    self.reset_pid()
                    self.state=FB_DRIVING
                    rospy.loginfo(
                        f"SCAN_FALLBACK -->FB_DRIVING  |"
                        F" Riktning 0 {math.degrees(self.theta_avoid):.1f}"
                    )
                elif rotated_so_far >= FALLBACK_MAX_ROTATE:
                    # Ingen fri väg hittad alls – återgå till IDLE
                    rospy.logwarn("SCAN_FALLBACK: ingen fri väg -> IDLE")
                    self.obstacle_pub.publish(Bool(data=False))
                    self.camera_active = False
                    self.state= IDLE
                else: 
                    # Fortsätt rotera åt fallback_scan_dir
                    step_target = self.normalize_angle(self.current_theta + self.fallback_scan_dir * FALLBACK_SCAN_STEP)
                    self.twist_pub.publish(Twist2DStamped(v=0.0, omega=self.fallback_scan_dir * abs(self.pid_steer(step_target))))
                    rospy.loginfo_throttle(
                        1.0,
                        f"SCAN_FALLBACK | ToF={self.tof_range:.2f}m | "
                        f"roterat={math.degrees(rotated_so_far):.1f}gradder"
                    )
            # Kör 35 cm i theta_avoid-riktning, sedan RETURNING
            elif self.state ==FB_DRIVING:
                self.fallback_drive_distance += AVOID_VELOCITY *dt_avoid
                if self.fallback_drive_distance>=FALLBACK_DRIVE:
                    self.stop()
                    rospy.sleep(0.2)
                    self.reset_pid()
                    self.state=RETURNING
                    rospy.loginfo(
                        f"FB_DRIVING -->> RETURNING | "
                        f"kört={self.fallback_drive_distance:.2f}m"
                    )
                else:
                    if self.tof_range < TOF_WALL_DIST:
                        rospy.loginfo("FB_DRIVING:   VÄGG---->> REVERSING")
                        self.start_reversing()
                    else:
                        omega = self.pid_steer(self.theta_avoid)
                        self.twist_pub.publish(Twist2DStamped(v=AVOID_VELOCITY, omega=omega))
                        rospy.loginfo_throttle(
                            2.0,
                            f"FB_DRIVING | {self.fallback_drive_distance:.2f}/{FALLBACK_DRIVE:.2f}m"
                        )

            elif self.state == ROTATING:        # Om tillstånd är ROTATING
                if self.tof_range < TOF_WALL_DIST:
                    self.stop()
                    rospy.loginfo_throttle(
                        1.0,
                        f"ROTATING väntar: ToF={self.tof_range:.2f}m (hinder framför)"
                    )
                elif self.rotate_to(self.theta_avoid):
                    self.total_rotation_done = self.normalize_angle(self.current_theta - self.theta_at_rotation_start)
                    self.distance_driven = 0.0
                    self.reset_pid()
                    self.state = AVOIDING
                    rospy.loginfo(
                        f"ROTATING -> AVOIDING | "
                        f"roterat={math.degrees(self.total_rotation_done):.1f}grader | "
                        f"ska köra {self.avoid_distance:.2f}m"
                    )
                else:
                    rospy.loginfo_throttle(
                        1.0,
                        f"ROTATING | fel={math.degrees(self.angle_error_to(self.theta_avoid)):.1f}grader"
                    )
               # if self.rotate_to(self.theta_avoid):      # Roterar mot undvikande riktning
                #    self.total_rotation_done =self.normalize_angle(self.current_theta -self.theta_at_rotation_start)
                 #   self.distance_driven= 0.0
                  #  self.reset_pid()
                  #  self.state =AVOIDING
                  # rospy.loginfo(
                  #      f"ROTATING --->>> AVOIDING | "
                   #     f"roterat={math.degrees(self.total_rotation_done):.1f}grader"
                   #     f"ska köra {self.avoid_distance:.2f}m"
                  #  )
                #else:
                #    rospy.loginfo_throttle(1.0,
                #       f"ROTATING | fel={math.degrees(self.angle_error_to(self.theta_avoid)):.1f} grader")
            elif self.state == AVOIDING:
                if self.latest_image is None:
                    rate.sleep(); continue
                
                if self.distance_driven >= self.avoid_distance:
                    self.stop()
                    rospy.sleep(0.25)
                    self.reset_pid()
                    self.state= RETURNING
                    rospy.loginfo(f"AVOIDING -->>> RETURNING |"
                                 f"kört={self.distance_driven:.2f}/{self.avoid_distance:.2f}m"
                    )
                else:
                    # Kolla efter vägg via kamera (ToF-vägg hanteras i callback)
                    found, _, is_wall, _ = self.analys_image(self.latest_image)
                    if is_wall or self.tof_range<TOF_WALL_DIST:
                        self.stop()
                        self.start_reversing()
                        rospy.loginfo("AVOIDING -->> REVERSING (vägg under körning)")
                     #   self.reverse_start = now
                      #  self.reset_pid()
                      #  self.state =REVERSING
                      #  rospy.loginfo("AVOIDING → REVERSING (vägg via kamera)")
                    else:
                        omega = self.pid_steer(self.theta_avoid)
                        self.twist_pub.publish(Twist2DStamped(v=AVOID_VELOCITY, omega=omega))
                        self.distance_driven += AVOID_VELOCITY * dt_avoid # Uppdatera hur långt vi kört (tid * hastighet)
                        rospy.loginfo_throttle(2.0,
                            f"AVOIDING | "
                            f"{self.distance_driven:.2f}/{self.avoid_distance:.2f}m | "
                            f"omega={omega:.2f}"
                        )
            elif self.state == REVERSING:
                elapsed = now - self.reverse_start
                if elapsed < REVERSE_DURATION:
                    self.twist_pub.publish(Twist2DStamped(v=-REVERSE_SPEED, omega=0.0))
                    rospy.loginfo_throttle(
                        1.0, f"REVERSING | {elapsed:.1f}/{REVERSE_DURATION:.1f}s"
                    )
        
        #        if (now - self.reverse_start) < REVERSE_DURATION:    # Om backtiden inte är klar
         #           self.twist_pub.publish(Twist2DStamped(v=-AVOID_VELOCITY, omega=0.0))  # Backa rakt bakåt
          #          rospy.loginfo_throttle(1.0,
           #             f"REVERSING | "
            #            f"{(now-self.reverse_start):.1f}/{REVERSE_DURATION:.1f}s"
             #       )
                else:   # När backning är klar
                    self.stop()
                    rospy.sleep(0.25)
                    self.reset_pid()
                    self.scan_results= []
                    self.scan_start= now
                    self.state= SCANNING
                    rospy.loginfo("REVERSING -->>>> SCANNING")

            elif self.state == RETURNING:
                if self.tof_range < TOF_WALL_DIST:
                    rospy.loginfo("RETURNING: vägg -> REVERSING")
                    self.start_reversing()
                    rate.sleep(); continue
                if self.tof_range < TOF_THRESHOLD and self.latest_image is not None:  # Om nytt hinder är nära
                    found,_,_,_ =self.analys_image(self.latest_image)
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
                    rospy.loginfo(
                        f"RETURNING -->> IDLE | "
                        f"theta_orig={math.degrees(self.theta_original):.1f}"
                    )
                else:
                    rospy.loginfo_throttle(1.0,
                        f"RETURNING | "
                        f"fel={math.degrees(self.angle_error_to(self.theta_original)):.1f}°"
                    )                   

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