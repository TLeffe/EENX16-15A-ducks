#!/usr/bin/env python3

import os
import cv2       #  används för att bearbeta kamerabilder
import math
import rospy
import yaml
import threading
import numpy as np
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import Range, CompressedImage
from std_msgs.msg import Bool,Float32

#------Avståndsgränser
TOF_THRESHOLD = 0.40           # meter - triggar kameran om föremål är närmare än detta
TOF_WALL_DIST = 0.17   # meter - under detta = vägg, backa direkt utan kamera
TOF_CLEAR_DIST  = 0.50   # meter - används i ToF-fallback scan
FALLBACK_DRIVE  = 0.35   # meter - hur långt vi kör i fallback-läge
MAX_TOF_FOR_DISTANCE = 0.80   # m - skyddar Pythagoras mot 100m-värden 
MIN_TOF_VALID        = 0.02   # m - under detta = brus, ignorera

#-----Hastigheter-------
AVOID_VELOCITY = 0.15            #  hastighet under undvikande
REVERSE_SPEED = 0.15           #  # m/s bakåt
FALLBACK_ROTATE_SPEED = 0.7   # rad/s konstant omega vid fallback-scan (enklare än PID)
REVERSE_DISTANCE = 0.20 # m 
REVERSE_DURATION = REVERSE_DISTANCE / REVERSE_SPEED     # Tid för att backa 20 cm

#---- camera------
MAX_AVOID_ANGLE = math.radians(30)       # Max vinkel för undvikande i radianer
CAMERA_HFOV = 160.0             # grader - kamerans horisontella synfält på Duckiebot
MIN_CONTOUR_AREA = 800      # Minsta area för att räkna som objekt
WALL_THRESHOLD = 50
SCAN_FRAMES = 4                      # Antal bilder att samla in vid scanning
SCAN_TIMEOUT = 1.5               # Max tid för scanning i sekunder

# ----- PID--
KP_THETA = 2               # proportionell — hur hårt vi styr mot rätt riktning
KI_THETA =  0.05          # integral — kompenserar konstant drift
KD_THETA = 0.08
OMEGA_MAX = 1.3              # max vridningshastighet (säkerhetsgräns)

# ---- vinkeltoleranser 
ANGLE_THRESHOLD  = math.radians(5)   # Vinkelgräns för att anses vara framme
ROTATING_EXTRA_STEP   = math.radians(8)   # rad - stegstorlek vid ToF-fallback scan
FALLBACK_MAX_ROTATE  = math.radians(170) # rad - max rotation under fallback scan
FALLBACK_MAX_PER_DIR = math.radians(80)

#---- minimiköravstånd 
MIN_AVOID_DISTANCE = 0.25   # meter mi-nsta köravstånd
MAX_AVOID_DISTANCE = 1.20   # m - övre gräns (skydd om Pythagoras ger orimligt värde)
MAX_AVOID_TIME = 5.0       # s - max tid i AVOIDING fastnar
STATE_TIMEOUT = 10.0


# --- BEV
BEV_W = 400      # BEV-bildens bredd (efter transformation)
BEV_H = 300      # BEV-bildens höjd (efter transformation)


#  Kamera-intrinsics (från er kalibreringsfil) 
# Används för att rätta linsförvrängning innan BEV-transformation.
CAMERA_MATRIX = np.array([
    [309.34488578183976, 0.0,318.97431157284797],
    [0.0, 324.3079313761394,249.74718090589798],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DIST_COEFFS = np.array([
    -0.26850619979326934,
     0.14095182385200886,
     0.029986662850901284,
     0.000792436879591894,
    -0.03422901917827191
], dtype=np.float64)

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

        #   sensor 
        self.tof_range = float ('inf')     # Senaste avståndet från ToF
        self.latest_image = None
        self.camera_active = False         # True när kameran analyserar
        self.state = IDLE            # State machine — börjar i IDLE
        self.wall_detected = False
        self.tof_initialized = False

        # Vinklar
        self.theta_original          = None  # vinkeln mot målet - fryses i IDLE
        self.current_theta           = 0.0  # robotens faktiska vinkel (odometri)
        self.theta_avoid             = 0.0  # bestäms i SCANNING, låst sedan
        self.theta_at_rotation_start = 0.0  # för att spåra hur mycket vi roterat
        self.total_rotation_done     = 0.0  # hur många radianer vi roterat totalt
        self.avoid_turn_sign         = 1.0  # Svängriktning: +1.0 = vänster, -1.0 = höger.
        
        # Köravstånd (Pythagoras)
        self.avoid_distance  = MIN_AVOID_DISTANCE
        self.distance_driven = 0.0
        self.avoid_start_time = 0.0
        # scanning        
        self.scan_results= []        # Lista med undvikande vinklar
        self.scan_start= 0.0
        
        # backnin
        self.reverse_start  = 0.0    # Starttid för backning

        self.fallback_scan_dir = -1.0   # -1 = höger, +1 = vänster
        self.fallback_scan_start_theta = 0.0  # vinkel när fallback-scan startade
        self.fallback_drive_distance   = 0.0
        self.fallback_total_rotated = 0.0
        self.fallback_prev_theta = 0.0
        self.fallback_rotated_per_dir = 0.0

        self._integral = 0.0              #   PID-sate
        self._prev_error = 0.0


        #  Undistortion-karta (beräknas EN gång, snabb sedan) 
        # cv2.initUndistortRectifyMap ger pixelkartor -> remap är ungefär 3 ms
        h, w = 480, 640
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            CAMERA_MATRIX, DIST_COEFFS, None, CAMERA_MATRIX, (w, h), cv2.CV_16SC2)
        rospy.loginfo("Undistortion-karta beräknad")

        # Homografi
        self.M = self.load_homography()
        if self.M is None:
            rospy.logwarn("Homografi saknas - använder default-homografi")
            self.M = self.default_homography()
        rospy.loginfo(f"Homografi laddad:\n{self.M}")
        
        self.obstacle_pub = rospy.Publisher(self.obstacle_topic, Bool, queue_size=1)    #  Publisher som skickar sant/falskt om hinder finns
        self.twist_pub = rospy.Publisher(self.twist_topic, Twist2DStamped, queue_size=1)  # Publisher som skickar hastighetskommandon till roboten
        
        rospy.Subscriber(self.tof_topic, Range, self.callback_tof)      # Lyssnar på ToF-sensorn och kör callback_tof vid nytt meddelande
        rospy.Subscriber(self.camera_topic, CompressedImage, self.callback_camera)   # Lyssnar på kameran och kör callback_camera vid ny bild
        rospy.Subscriber(self.desired_theta_topic, Float32, self.callback_desired_theta)
        rospy.Subscriber(self.current_theta_topic, Float32,self.callback_current_theta)

        rospy.loginfo(f"ObstacleDetectionNode startad")

    def load_homography(self):
        """
        Laddar homografi från Duckietowns extrinsic-kalibreringsfil.
        Sökväg: /data/config/calibrations/camera_extrinsic/<VEHICLE_NAME>.yaml
        Format: homography: [h11,h12,h13, h21,h22,h23, h31,h32,h33]  (9 värden)
        """
        path = f"/data/config/calibrations/camera_extrinsic/{self.vehicle_name}.yaml"

        if not os.path.exists(path):
            rospy.logwarn(f"Kalibreringsfil saknas: {path}")
            return None

        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)

            if 'homography' not in data:
                rospy.logwarn(f"Nyckel 'homography' saknas i {path}")
                return None

            H_list = data['homography']
            if len(H_list) != 9:
                rospy.logwarn(f"homography har {len(H_list)} värden, förväntade 9")
                return None

            # reshape(3,3) ger rätt rad-major ordning direkt från YAML-listan
            H = np.array(H_list, dtype=np.float64).reshape(3, 3)
            rospy.loginfo(f"Homografi laddad från {path}")
            return H

        except Exception as e:
            rospy.logerr(f"Fel vid läsning av homografi: {e}")
            return None

    def default_homography(self):
        """
        Fallback-homografi om kalibreringsfilen saknas.
        Baserad på typiska Duckiebot-mått (kameran 10 cm över golvet, ungefär 15° nedåt).
        """
        src = np.float32([
            [200, 180], [440, 180],
            [560, 380], [ 80, 380]
        ])
        dst = np.float32([
            [ 80,   0], [320,   0],
            [320, BEV_H], [ 80, BEV_H]
        ])
        return cv2.getPerspectiveTransform(src, dst)
        
    def callback_desired_theta (self, msg):
        if self.state == IDLE:               # Behåll originalvinkeln fryst under undvikande
            if self.theta_original is None:
                rospy.loginfo(f"theta_original initierad: {math.degrees(msg.data):.1f}grader")
            self.theta_original= msg.data     # Under AVOIDING/RETURNING ska den INTE skrivas över — vi vill minnas vart vi skulle

    def callback_current_theta(self, msg):   # Tar emot robotens aktuella vinkel för PID-reglering
        self.current_theta = msg.data
        
    def callback_tof(self, msg):   #  Körs varje gång ToF-sensorn skickar ett nytt avstånd.
        if msg.range < MIN_TOF_VALID or msg.range > 5.0:
            return
        
        self.tof_range = msg.range      # Sparar senaste avståndet
        self.tof_initialized = True

        if msg.range < TOF_WALL_DIST and self.state != REVERSING:         # Kolla om avståndet är mindre än 0.17
            rospy.loginfo(f"ToF VÄGG: {msg.range:.2f}m < {TOF_WALL_DIST}m --->>> backar direkt")
            self.wall_detected = True
            return
        if msg.range < TOF_WALL_DIST and self.state != REVERSING:
           # if not self.camera_active:          # kolla om kameran inte redan är aktiv
            self.camera_active = True
            rospy.loginfo(f"TOF: Formål på {msg.range:.2f}m ----> aktivera kamera")
            return
        
        if msg.range < TOF_THRESHOLD:
            if not self.camera_active:
                self.camera_active = True
        else:
            if self.state == IDLE:             # Om avståndet är större än tröskeln
                rospy.loginfo(f"ToF: Vägen klar ({msg.range:.2f}m) ------> kamera inaktiv")
                self.camera_active = False     # Inget hinder nära --> stäng av kamera-analys
                self.obstacle_pub.publish(Bool(data=False))  

    def callback_camera(self, msg):     #   Körs varje gång kameran skickar en ny bild.
        if not self.camera_active and self.state == IDLE:      # Hoppa över om kameran inte är aktiv
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
        if target is None:
            return 0.0 
        return self.normalize_angle(target - self.current_theta)    # Returnerar negativt = vrida höger, positivt = vrida vänster
    
    def reset_pid(self): 
        self._integral = 0.0
        self._prev_error = 0.0

    def pid_steer(self, target, dt=0.1):
        if target is None or dt <= 0:
            return 0.0
        error = self.angle_error_to(target)
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        self._integral  = max(-3.0, min(3.0, self._integral + error * dt))
        omega = KP_THETA * error + KI_THETA * self._integral + KD_THETA * derivative
        omega = max(-OMEGA_MAX, min(OMEGA_MAX, omega))
        return omega

    def rotate_to(self, target, dt):        # Roterar mot en given vinkel
        if target is None:
            self.stop()
            return True        
        if abs(self.angle_error_to(target)) < ANGLE_THRESHOLD:    # Om nära nog
            self.stop()
            rospy.loginfo(f"Nu ska jag rotera mot given vinkel :::)))")
            return True

        self.twist_pub.publish(Twist2DStamped(v=0.0, omega=self.pid_steer(target, dt)))   # Roterar på plats
        return False
    
    def start_reversing(self):
        self.stop()
        self.reverse_start= rospy.Time.now().to_sec()
        self.reset_pid()
        self.wall_detected = False
        self.state = REVERSING
        self.state_entry_time = rospy.Time.now().to_sec()
        rospy.loginfo("--->>> REVERSING")

    def compute_avoid_distance(self, tof_dist, edge_angle_deg ):    # använder av Pythagoras
        if not self.tof_initialized:
            tof_dist = TOF_THRESHOLD        
        a = min(max(tof_dist, 0.05), MAX_TOF_FOR_DISTANCE) # a = avstånd till hinder (begränsat till MAX_TOF_FOR_DISTANCE)
        b = a * math.tan(math.radians(min(abs(edge_angle_deg), 89.0)))   # Begränsa vinkel till < 89° (tan(90) = oändligt)
        c = math.sqrt(a**2 + b**2) + 0.10   #  10 cm exgtra 
        result = max(MIN_AVOID_DISTANCE, min(MAX_AVOID_DISTANCE, c))
        rospy.loginfo(
            f"Pythagoras: a={a:.2f}m | b={b:.2f}m | "
            f"c={c:.2f}m --->> kör {result:.2f}m"
        )
        return result

    def check_state_timeout(self, now):
        """Watchdog: om ett tillstånd hänger > STATE_TIMEOUT → återgå till IDLE."""
        if self.state != IDLE and (now - self.state_entry_time) > STATE_TIMEOUT:
            rospy.logwarn(f"Watchdog: {self.state} hängde > {STATE_TIMEOUT}s → IDLE")
            self.stop()
            self.camera_active = False
            self.wall_detected = False
            self.obstacle_pub.publish(Bool(data=False))
            self.state            = IDLE
            self.state_entry_time = now
            return True
        return False
    
    def _undistort(self, image):  # Rättar linsförvrängning med förberäknad pixelkarta.
        return cv2.remap(image, self._map1, self._map2, cv2.INTER_LINEAR)

    def to_bev (self, image):
       #  Transformerar kamerabild till Bird's Eye View (ovanifrån).
        try:
            undistorted = self._undistort(image)
            return cv2.warpPerspective(undistorted, self.M, (BEV_W, BEV_H))
        except Exception as e:
            rospy.logerr_throttle(3.0, f"BEV misslyckades: {e}")
            return cv2.resize(image, (BEV_W, BEV_H))
        
    def remove_floor(self, bev_gray):  #       Tar bort golvets textur med adaptiv tröskling.
        mask = cv2.adaptiveThreshold(      #  # Adaptiv tröskling - jämför varje pixel med lokalt medelvärde
            bev_gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,  # Stort grannskap = ignorerar globala ljusvariationer
            C=10           # Subtrahera 10 från medelvärdet
        )

        # Morfologisk öppning = ta bort små bruspunkter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  
    

    def analys_image (self, image):   # analysera  komerabild för att bekräfta föremål. 
        if image is None:   #Returnerar: (found, theta_avoid, is_wall, edge_angle_deg, turn_sign)
            return False, 0.0  , False , 0.0 , 1.0

        # --- Steg 1: Bird's Eye View 
        bev = self.to_bev(image)   
         # -- Steg 2: Förbearbeta bilden 
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)   # Gör bilden gråskalig/ COLOR_BGR2GRAY konverterar bilden från BEG TO GRAY / cv2.cvtColor --> ändra färgeformat på en bild 
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Suddar lite för att minska brus/(5, 5)  lagom blur vi har 0 då vi låter opencv välja bästa styrkan auto
        
        # -- Steg 3: Ta bort golvet 
        floor_removed = self.remove_floor(blurred)       
        edges = cv2.Canny(floor_removed, 30, 100)        # Hittar kanter i bilden,  <50 → ignorera, 50–150 → kanske viktigt, 150 → definitivt viktig / (50-150) --> ta bara tydliga kanter, men tillåt lite svagare kant
        contours, _  = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # hittar kanter i bilden
        '''cv2.findContours()--> tar en edge-bild, hittar alla objekt, returnerar deras former
        cv2.RETR_EXTERNAL --> tar bara yttersta konturer/ cv2.CHAIN_APPROX_SIMPLE--> spara bara vikiga punker'''   
        height, width = bev.shape[:2]
        center_x = width // 2
        roi_bottom = int(height * 0.7)  # Bara övre 70% = framåt
        best_contour = None       
        best_area = 0

        for contour in contours:   
            area = cv2.contourArea(contour)   # Storlek på objektet
            if area < MIN_CONTOUR_AREA: # Ignorera små konturer (brus)
                continue

            # Hämtar en rektangel runt konturen
            x , y,w,h = cv2.boundingRect(contour)    # Lätt att hitta mitten av objektet
         #   contour_center_x = x + w // 2      # konturens mittpunkt x
            contour_center_y = y + h // 2      # konturens mittpunkt y
            
            # Kriterier: i framåt-zonen, inom 40% från mitten, störst area
            if contour_center_y < roi_bottom and area > best_area:
                best_area = area
                best_contour = contour            

        if best_contour is None:   # Inga hinder hittades i mittzon

            return False, 0.0 , False , 0.0 , 1.0
        
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
            rospy.loginfo_throttle(2.0, f"VÄGG från kamera bild  |  vänster={left_space} | höger={right_space} --> backar")
            return True, 0.0, True, 0.0 , 1.0
        
        if left_space > right_space:
            edge_pixel = x
            side_text = "VÄNSTAR"
            turn_sign = 1.0
            rospy.loginfo("Mest plats på VÄNSTAR --->>> svänger vänstar")
        else:
            edge_pixel = x + w   # hindrets högra kant
            side_text = "HÖGER"
            turn_sign = -1.0     # negativ omega = höger
            rospy.loginfo("Mest plats på HÖGER --->>> svänger höger")
        

        pixel_offset = (edge_pixel- width/2.0) /(width/2.0) * (CAMERA_HFOV /2)  # Pixel till vinkel
        angle_error = max(-MAX_AVOID_ANGLE, min(MAX_AVOID_ANGLE, math.radians(pixel_offset)))    # Begränsa vinkel
        theta_avoid = self.normalize_angle(self.current_theta + angle_error)       # Beräkna undvikande riktning
        edge_angle_deg = math.degrees(abs(angle_error))  #Konverterar vinkeln från radianer till grader och gör den positiv.


        rospy.loginfo_throttle(
            3.0,
            f"Hinder | sida={side_text} | vänster={left_space}px | "
            f"höger={right_space}px | theta_avoid={math.degrees(theta_avoid):.1f}| "
            f"kant-vinkel={edge_angle_deg:.1f}grader"
        )


        return True, theta_avoid, False, edge_angle_deg , turn_sign
    
    def run(self):
        rate = rospy.Rate(20)
        last_time = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()    # Hämtar nuvarande ROS-tid i sekunder
            dt  = max(0.001, min(0.1, now - last_time))
            last_time = now

            if self.check_state_timeout(now):
                rate.sleep()
                continue     

            if self.wall_detected and self.state != REVERSING:
                rospy.logwarn(f"wall_detected=True i [{self.state}] --->>> backar")
                self.obstacle_pub.publish(Bool(data=True))
                self.start_reversing()
                rate.sleep()
                continue

            if self.theta_original is None:
                rate.sleep()
                continue

            if self.state ==IDLE:   # IDLE--->> väntar på hinder
                if self.camera_active and  self.latest_image is not None :  # Om kamera är aktiv och bild finns
                    self.obstacle_pub.publish(Bool(data=True))    # Publicera att hinder finns
                    self.stop()
                    self.scan_results= []     # Töm gamla scanresultat
                    self.scan_start = now     # Spara scan-starttid
                    self.state = SCANNING
                    self.state_entry_time = now
                    rospy.loginfo(f"IDLE--->SCANNING | theta_orig={math.degrees(self.theta_original):.1f}")
            
            elif self.state== SCANNING:
                self.stop()
                if self.latest_image is not None:     # Om vägg upptäcks
                    found, theta, is_wall, edge_deg, turn_sign = self.analys_image(self.latest_image)             
                    
                    if found and is_wall:
                        rospy.loginfo("SCANNING --> REVERSING (vägg via BEV-kamera)")
                        self.start_reversing()
                        rate.sleep(); continue     # Vänta och hoppa till nästa loop
                    if found:
                        safe_tof = min(self.tof_range if self.tof_initialized else TOF_THRESHOLD, MAX_TOF_FOR_DISTANCE)
                        self.scan_results.append((theta, edge_deg, safe_tof, turn_sign))     # Spara undvikande vinkel
            
                ready = len(self.scan_results) >= SCAN_FRAMES     # Kolla om tillräckligt många bilder samlats
                timeout = (now -self.scan_start) >= SCAN_TIMEOUT   # Kolla om tiden tagit slut

                if ready or (timeout and self.scan_results):       # Om redo eller timeout med resultat
                    thetas    = [r[0] for r in self.scan_results]  # Plockar ut ALLA första värden (theta-vinklar)
                    edge_degs = [r[1] for r in self.scan_results]  #  Plockar ut ALLA andra värden (kantvinklar i grader)
                    tof_vals  = [r[2] for r in self.scan_results]  # Plockar ut ALLA tredje värden (ToF-avstånd)
                    turn_signs = [r[3] for r in self.scan_results]
                    
                    self.theta_avoid = float(np.median(thetas))      # Ta median av vinklarna
                    median_egde_deg = float(np.median(edge_degs)) 
                    median_tof = float(np.median(tof_vals))  
                    # Majoritetsbeslut: om fler bilder pekade åt vänster → vänster
                    self.avoid_turn_sign = 1.0 if sum(turn_signs) >= 0 else -1.0
                    
                    self.avoid_distance = self.compute_avoid_distance(median_tof, median_egde_deg)  # # Beräkna körsträcka och spara startvinkel
                    self.theta_at_rotation_start = self.current_theta
                    self.distance_driven = 0.0
                    self.reset_pid()
                    self.state= ROTATING
                    self.state_entry_time = now
                    rospy.loginfo(f"SCANNING --> ROTATING |"
                    f"theta_avoid={math.degrees(self.theta_avoid):.1f} | "
                    f"sign={self.avoid_turn_sign:+.0f} | "
                    f"köravstånd={self.avoid_distance:.2f}m"
                    )
                elif timeout:      # Om timeout utan resultat
                    rospy.loginfo("SCANNING -->>> SCAN_FALLBACK (kamera hittade inget, använder ToF)")
                    self.fallback_scan_start_theta = self.current_theta
                    self.fallback_scan_dir = -1.0  # Börja med att rotera åt höger
                    self.fallback_total_rotated = 0.0
                    self.fallback_prev_theta = self.current_theta
                    self.reset_pid()
                    self.state=SCAN_FALLBACK
                    self.state_entry_time = now

            elif self.state ==SCAN_FALLBACK:
                delta = abs(self.normalize_angle(self.current_theta - self.fallback_prev_theta))
                self.fallback_total_rotated += delta
                self.fallback_rotated_per_dir += delta
                self.fallback_prev_theta     = self.current_theta
            
                if self.tof_range > TOF_CLEAR_DIST:
                    self.theta_avoid = self.current_theta
                    self.fallback_drive_distance = 0.0
                    self.reset_pid()
                    self.state=FB_DRIVING
                    self.state_entry_time = now
                    rospy.loginfo(
                        f"SCAN_FALLBACK -->FB_DRIVING  |"
                        F" Riktning 0 {math.degrees(self.theta_avoid):.1f}"
                    )
                elif self.fallback_rotated_per_dir >= FALLBACK_MAX_PER_DIR and self.fallback_scan_dir == -1.0:   #  prova vänster
                    # Inget åt höger - prova vänster istället
                    rospy.logwarn("SCAN_FALLBACK: ingen  år höger -> provar vänstar")
                    self.fallback_scan_dir = 1.0
                    self.fallback_scan_start_theta = self.current_theta
                    self.fallback_rotated_per_dir  = 0.0
                    self.stop()

                elif self.fallback_total_rotated >= FALLBACK_MAX_ROTATE:
                    rospy.logwarn("SCAN_FALLBACK: ingen!!! FRI VÄGG-> IDLE ")
                    self.obstacle_pub.publish(Bool(data=False))
                    self.camera_active = False
                    self.state= IDLE
                    self.state_entry_time = now
                else: 
                    # Fortsätt rotera åt fallback_scan_dir
                    #step_target = self.normalize_angle(self.current_theta + self.fallback_scan_dir * FALLBACK_SCAN_STEP)
                    self.twist_pub.publish(Twist2DStamped(v=0.0, omega=self.fallback_scan_dir * FALLBACK_ROTATE_SPEED))
                    rospy.loginfo_throttle(
                        3.0,
                        f"SCAN_FALLBACK | ToF={self.tof_range:.2f}m | "
                        f"riktning={'höger' if self.fallback_scan_dir < 0 else 'vänster'} | "
                        f"roterat={math.degrees(self.fallback_total_rotated):.1f}gradder"
                    )
            # Kör 35 cm i theta_avoid-riktning, sedan RETURNING
            elif self.state ==FB_DRIVING:
                self.fallback_drive_distance += AVOID_VELOCITY *dt
                if self.fallback_drive_distance>=FALLBACK_DRIVE:
                    self.stop()
                    self.reset_pid()
                    self.state=RETURNING
                    self.state_entry_time = now
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
                if self.rotate_to(self.theta_avoid, dt):
                    self.total_rotation_done = self.normalize_angle(self.current_theta - self.theta_at_rotation_start)
                    still_blocked = False
                    if self.latest_image is not None and self.tof_range < TOF_THRESHOLD:
                        found, _, _, _, _ = self.analys_image(self.latest_image)
                        still_blocked = found
                    if still_blocked:
                        self.theta_avoid = self.normalize_angle(self.theta_avoid + self.avoid_turn_sign * ROTATING_EXTRA_STEP)
                        rospy.loginfo(
                            f"ROTATING: framför fortfarande blockerad  "
                            f"justerar theta_avoid till {math.degrees(self.theta_avoid):.1f}grader "
                            f"(sign={self.avoid_turn_sign:+.0f})"
                        )
                    else: 
                        self.distance_driven = 0.0
                        self.avoid_start_time = now
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
            elif self.state == AVOIDING:
                if (now - self.avoid_start_time) > MAX_AVOID_TIME:
                    rospy.logwarn("AVOIDING timeout → RETURNING")
                    self.stop()
                    self.reset_pid()
                    self.state= RETURNING
                    self.state_entry_time = now

                elif self.distance_driven >= self.avoid_distance:
                    self.stop()
                    self.reset_pid()
                    self.state = RETURNING
                    self.state_entry_time = now
                    rospy.loginfo(f"AVOIDING -->>> RETURNING |"
                                 f"kört={self.distance_driven:.2f}/{self.avoid_distance:.2f}m"
                    )
                else:

                    if self.tof_range < TOF_WALL_DIST:
                        rospy.loginfo("AVOIDING: vägg → REVERSING")
                        self.start_reversing()
                    else:
                        omega = self.pid_steer(self.theta_avoid)
                        self.twist_pub.publish(Twist2DStamped(v=AVOID_VELOCITY, omega=omega))
                        self.distance_driven += AVOID_VELOCITY * dt # Uppdatera hur långt vi kört (tid * hastighet)
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
                else:   # När backning är klar
                    self.stop()
                    self.reset_pid()
                    self.scan_results= []
                    self.scan_start= now
                    self.state= SCANNING
                    self.state_entry_time = now
                    rospy.loginfo("REVERSING -->>>> SCANNING")

            elif self.state == RETURNING:
                if self.tof_range < TOF_THRESHOLD and self.latest_image is not None:
                    found, _, _, _, _ = self.analys_image(self.latest_image)
                    if found:
                        self.scan_results = []
                        self.scan_start = now
                        self.stop()
                        self.state = SCANNING
                        self.state_entry_time = now
                        rospy.loginfo("RETURNING -> SCANNING (nytt hinder)")
                        rate.sleep();continue
                    
                if self.rotate_to(self.theta_original,dt):
                    self.obstacle_pub.publish(Bool(data=False))
                    self.camera_active = False
                    self.state = IDLE
                    self.state_entry_time = now
                    rospy.loginfo(
                        f"RETURNING -> IDLE | "
                        f"theta_orig={math.degrees(self.theta_original):.1f}grader")
                else:
                    rospy.loginfo_throttle(1.0,
                        f"RETURNING | fel={math.degrees(self.angle_error_to(self.theta_original)):.1f}grader")              

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