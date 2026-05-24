#!/usr/bin/env python3
"""
Systemöversikt:
    1. ToF-sensorn triggar kameraanalysen när ett objekt är nära.
    2. Kamerabilden transformeras till Bird's Eye View (BEV) och analyseras.
    3. Roboten roterar mot den sida där det finns mest fri yta.
    4. Roboten kör förbi hindret och återgår till ursprunglig riktning.
    5. Om kameran inte hittar något används ToF-baserad fallback-scanning.
"""
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

# =========================
# Avståndsgränser
# =========================
TOF_THRESHOLD = 0.40              # [m] Aktiverar kameran när ett objekt är närmare än detta
TOF_WALL_DIST = 0.17              # [m] under detta = vägg, backa direkt utan kamera
TOF_CLEAR_DIST  = 0.50            # [m] Avstånd som räknas som fri väg vid fallback-scanning
FALLBACK_DRIVE  = 0.35            # [m] Hur långt vi kör i fallback-läge
MAX_TOF_FOR_DISTANCE = 0.80       # [m] Begränsar orimliga ToF-värden i avståndsberäkningen
MIN_TOF_VALID        = 0.02       # [m] under detta = brus, ignorera

# =========================
# Hastigheter
# =========================
AVOID_VELOCITY = 0.15             # [m/s] Hastighet under undvikande
REVERSE_SPEED = 0.15              # [m/s] Bakåt
FALLBACK_ROTATE_SPEED = 0.7       # [rad/s] konstant omega vid fallback-scan (enklare än PID)
REVERSE_DISTANCE = 0.20           # [m]
REVERSE_DURATION = REVERSE_DISTANCE / REVERSE_SPEED     # [s] Tid för att backa  REVERSE_DISTANCE

# =========================
# Kamera
# =========================
MAX_AVOID_ANGLE = math.radians(30)      # Max vinkel för undvikande i radianer
CAMERA_HFOV = 160.0                     # [grader] kamerans horisontella synfält på Duckiebot
MIN_CONTOUR_AREA = 800                  # Minsta area för att räkna som objekt
WALL_THRESHOLD = 50                     # Minsta konturarea för att räknas som hinder
SCAN_FRAMES = 4                         # Antal bilder att samla in vid scanning
SCAN_TIMEOUT = 1.5                      # [s] Max tid för scanning i sekunder

# =========================
# PID-regulator
# =========================
KP_THETA = 2                            # Proportionell förstärkning
KI_THETA =  0.05                        # Integraldel - kompenserar konstant drift
KD_THETA = 0.08                         # Derivatadel - dämpar överskjutning
OMEGA_MAX = 1.3                         # [rad/s] Maximal tillåten vridhastighet

# =========================
# vinkeltoleranser  [rad]
# =========================
ANGLE_THRESHOLD  = math.radians(5)        # Robot anses riktad rätt när felet är mindre än detta
ROTATING_EXTRA_STEP   = math.radians(8)   # Extra vinkelsteg om vägen fortfarande är blockerad
FALLBACK_MAX_ROTATE  = math.radians(170)  # Max rotation under fallback scan
FALLBACK_MAX_PER_DIR = math.radians(80)   # Max rotation per riktning innan byte av håll

# =========================
# Köravstånd och timeouts
# =========================
MIN_AVOID_DISTANCE = 0.25                 # [m]  Minsta körsträcka vid hinderundvikning
MAX_AVOID_DISTANCE = 1.20                 # [m] övre gräns (skydd om Pythagoras ger orimligt värde)
MAX_AVOID_TIME = 5.0                      # [s]  max tid i AVOIDING fastnar
STATE_TIMEOUT = 6.0                       # [s] Watchdog-gräns för alla tillstånd

# =========================
# BEV-bildstorlek (efter transformation)
# =========================
BEV_W = 400      # [px] Bredd
BEV_H = 300      # [px] Höjd


# Kamera-intrinsics (från er kalibreringsfil) 
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

# =========================
# Tillstånd i tillståndsmaskinen
# =========================
IDLE          = "IDLE"             # Väntar på hinder
SCANNING      = "SCANNING"         # Kameraanalys pågår 
AVOIDING      = "AVOIDING"         # Roterar mot undvikningsriktning
ROTATING      = "ROTATING"         # Kör förbi hindret
REVERSING     = "REVERSING"        # Backar från vägg
RETURNING     = "RETURNING"        # Roterar tillbaka till ursprungsriktning
SCAN_FALLBACK = "SCAN_FALLBACK"    # ToF-rotering när kameran inte hittar hinder
FB_DRIVING    = "FB_DRIVING"       # Kör framåt i fallback-riktning



class ObstacleDetectionNode(DTROS):
    """
    ROS-nod för hinderdetektion och undvikning på Duckiebot.
 
    Kombinerar ToF-sensor och kamerabaserad BEV-analys i en tillståndsmaskin
    för att detektera, klassificera och undvika hinder i körvägen.
    """

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

        # sensor och tillståndsvariabler
        self.tof_range = float ('inf')       # Senaste ToF-mätning [m]
        self.latest_image = None             # Senaste kamerabild
        self.camera_active = False           # True när kameran analyserar
        self.state = IDLE                    # Aktuellt tillstånd
        self.wall_detected = False           # True när ToF mäter < TOF_WALL_DIST
        self.tof_initialized = False         # True efter första giltiga ToF-mätning

        # Vinklar och rotationsinformation
        self.theta_original          = None # vinkeln mot målet - fryses i IDLE
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
        self.scan_results= []                 # Insamlade detektionsresultat under SCANNING
        self.scan_start= 0.0                  # Tidsstämpel när SCANNING startade
        
        # backnin
        self.reverse_start  = 0.0             # Tidsstämpel när REVERSING startade
        
        # Fallback-scanning 
        self.fallback_scan_dir         = -1.0  # -1 = höger, +1 = vänster
        self.fallback_scan_start_theta = 0.0   # vinkel när fallback-scan startade
        self.fallback_drive_distance   = 0.0
        self.fallback_total_rotated    = 0.0
        self.fallback_prev_theta       = 0.0
        self.fallback_rotated_per_dir  = 0.0

        # PID-Tillstånd
        self._integral = 0.0                
        self._prev_error = 0.0

        # Undistortion-karta (beräknas en gång vid start)
        h, w = 480, 640
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            CAMERA_MATRIX, DIST_COEFFS, None, CAMERA_MATRIX, (w, h), cv2.CV_16SC2)
        rospy.loginfo("Undistortion-karta beräknad")

        # Homografi för omvandling från kamerabild till BEV
        self.M = self.load_homography()
        if self.M is None:
            rospy.logwarn("Homografi saknas - använder default-homografi")
            self.M = self.default_homography()
        rospy.loginfo(f"Homografi laddad:\n{self.M}")
        
        # Publishers
        self.obstacle_pub = rospy.Publisher(self.obstacle_topic, Bool, queue_size=1)      #  Publisher om hinder är akivt
        self.twist_pub = rospy.Publisher(self.twist_topic, Twist2DStamped, queue_size=1)  # Publisher hastighetskommandon till roboten
        
        # Subscribers 
        rospy.Subscriber(self.tof_topic, Range, self.callback_tof)                        # Prenumererar på ToF-sensorn
        rospy.Subscriber(self.camera_topic, CompressedImage, self.callback_camera)        # Prenumererar på komprimerade kamerabilder
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
        """
        Tar emot önskad målvinkel från twist_control_node.
        Vinkeln fryses när undvikning startar och uppdateras bara i IDLE.
        """
        if self.state == IDLE:               
            if self.theta_original is None:
                rospy.loginfo(f"theta_original initierad: {math.degrees(msg.data):.1f}grader")
            self.theta_original= msg.data     

    def callback_current_theta(self, msg): 
        """Tar emot robotens aktuella vinkel från odometri."""
        self.current_theta = msg.data
        
    def callback_tof(self, msg): 
        """
        Hanterar ToF-mätningar.
        Aktiverar kameraanalys när ett hinder är inom TOF_THRESHOLD.
        Triggar direkt backning om avståndet är under TOF_WALL_DIST.
        """
        if msg.range < MIN_TOF_VALID or msg.range > 5.0:
            return  # Ignorera ogiltiga mätningar
        
        self.tof_range       = msg.range
        self.tof_initialized = True
        if msg.range < TOF_WALL_DIST and self.state != REVERSING:
            self.camera_active = True
            rospy.loginfo(f"TOF: Formål på {msg.range:.2f}m - aktivera kamera")
            return
        
        if msg.range < TOF_THRESHOLD:
            if not self.camera_active:
                self.camera_active = True
        else:
            if self.state == IDLE:     
                rospy.loginfo(f"ToF: Vägen klar ({msg.range:.2f}m) - kamera inaktiv")
                self.camera_active = False 
                self.obstacle_pub.publish(Bool(data=False))  

    def callback_camera(self, msg):
        """
        Tar emot komprimerade kamerabilder och konverterar till OpenCV-format.
        Bilden sparas i self.latest_image för användning i kameraanalysen.
        """
        if not self.camera_active and self.state == IDLE:   
            return  # Spara inte bilder när kameran är inaktiv
        
        try: 
            np_arr = np.frombuffer(msg.data, np.uint8)  
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  
            if image is not None:
                self.latest_image = image
        except Exception as e: 
            rospy.logwarn(f"Kamerafel: {e}")

    #=========================
    # Rörelsekontrol
    #=========================

    def stop (self): 
        """Stoppar roboten omedelbart."""  
        self.twist_pub.publish(Twist2DStamped(v=0.0, omega=0.0))

    def normalize_angle(self, angle):    
        """Normaliserar vinkel till intervallet [-pi, pi]."""
        while angle >  math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle 
    
    def angle_error_to(self, target):
        """
        Beräknar minsta vinkelfel mot ett mål.
 
        Returns:
            float: Positivt = vrid vänster, negativt = vrid höger.
        """    
        if target is None:
            return 0.0 
        return self.normalize_angle(target - self.current_theta) 
    
    def reset_pid(self): 
        """Nollställer PID-tillståndet."""
        self._integral = 0.0
        self._prev_error = 0.0

    def pid_steer(self, target, dt=0.1):
        """
        Beräknar PID-styrsignal (omega) mot ett vinkelmål.
        Args:
            target: Målvinkel i radianer.
            dt:     Tidssteg i sekunder.
        Returns:
            float: Begränsad vinkelhastighet [rad/s].
        """
        if target is None or dt <= 0:
            return 0.0
        error = self.angle_error_to(target)
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        self._integral  = max(-3.0, min(3.0, self._integral + error * dt))

        omega = KP_THETA * error + KI_THETA * self._integral + KD_THETA * derivative
        omega = max(-OMEGA_MAX, min(OMEGA_MAX, omega))
        return omega

    def rotate_to(self, target, dt): 
        """
        Roterar roboten mot ett vinkelmål med PID-styrning.
        Returns:
            bool: True när målet är nått (fel < ANGLE_THRESHOLD).
        """        
        if target is None:
            self.stop()
            return True        
        if abs(self.angle_error_to(target)) < ANGLE_THRESHOLD: 
            self.stop()
            rospy.loginfo(f"Rotation klar - i önskade riktning.")
            return True

        self.twist_pub.publish(Twist2DStamped(v=0.0, omega=self.pid_steer(target, dt)))   # Roterar på plats
        return False
    
    def start_reversing(self):
        """Initierar backningstillståndet."""
        self.stop()
        self.reverse_start= rospy.Time.now().to_sec()
        self.reset_pid()
        self.wall_detected = False
        self.state = REVERSING
        self.state_entry_time = rospy.Time.now().to_sec()
        rospy.loginfo("-> REVERSING")

    def compute_avoid_distance(self, tof_dist, edge_angle_deg ): 
        """
        Beräknar körsträcka för hinderundvikning med Pythagoras sats.
        a = avstånd till hindret (ToF)
        b = sidoförflyttning (a x tan(kantvinkel))
        c = hypotenusa + 10 cm säkerhetsmarginal

        Returns:
            float: Begränsad körsträcka [m].
        """        
        if not self.tof_initialized:
            tof_dist = TOF_THRESHOLD        
        a = min(max(tof_dist, 0.05), MAX_TOF_FOR_DISTANCE) 
        b = a * math.tan(math.radians(min(abs(edge_angle_deg), 89.0)))   
        c = math.sqrt(a**2 + b**2) + 0.10   
        result = max(MIN_AVOID_DISTANCE, min(MAX_AVOID_DISTANCE, c))
        rospy.loginfo(
            f"Pythagoras: a={a:.2f}m | b={b:.2f}m | "
            f"c={c:.2f}m -> kör {result:.2f}m"
        )
        return result

    def check_state_timeout(self, now):
        """
        Watchdog: återgår till IDLE om ett tillstånd hänger längre än STATE_TIMEOUT.
        Returns:
            bool: True om timeout inträffade.
        """        
        if self.state != IDLE and (now - self.state_entry_time) > STATE_TIMEOUT:
            rospy.logwarn(f"Watchdog: {self.state} hängde > {STATE_TIMEOUT}s -> IDLE")
            self.stop()
            self.camera_active = False
            self.wall_detected = False
            self.obstacle_pub.publish(Bool(data=False))
            self.state            = IDLE
            self.state_entry_time = now
            return True
        return False
    
    #=========================
    # Bildbehandling
    #=========================

    def _undistort(self, image):
        """Korrigerar linsförvrängning med förberäknad pixelkarta."""
        return cv2.remap(image, self._map1, self._map2, cv2.INTER_LINEAR)

    def to_bev (self, image):
        """
        Transformerar kamerabild till Bird's Eye View med homografimatrisen.
 
        Returns:
            np.ndarray: BEV-bild av storlek (BEV_W x BEV_H).
        """
        try:
            undistorted = self._undistort(image)
            return cv2.warpPerspective(undistorted, self.M, (BEV_W, BEV_H))
        except Exception as e:
            rospy.logerr_throttle(3.0, f"BEV misslyckades: {e}")
            return cv2.resize(image, (BEV_W, BEV_H))
        
    def remove_floor(self, bev_gray):  
        """
        Tar bort golvets textur med adaptiv tröskling och morfologisk öppning.
        Args:
            bev_gray: Gråskalig BEV-bild.
        Returns:
            np.ndarray: Binär mask där golvpixlar är borttagna.
        """                
        mask = cv2.adaptiveThreshold(   
            bev_gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,  
            C=10   
        )
        # Morfologisk öppning = ta bort små bruspunkter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  
    
    def analys_image (self, image): 
        """
        Analyserar kamerabild för att identifiera hinder och bestämma undvikningsriktning.
 
        Pipeline:
            1. BEV-transformation
            2. Gråskalekonvertering och brusfiltrering (Gaussiskt filter)
            3. Borttagning av golvtextur (adaptiv tröskling)
            4. Kantdetektion (Canny)
            5. Konturanalys - hittar största hindret i framåt-ROI (övre 70%)
            6. Bestämmer svängriktning baserat på fritt utrymme
 
        Returns:
            tuple: (found, theta_avoid, is_wall, edge_angle_deg, turn_sign)
                found         (bool)  - hinder hittades
                theta_avoid   (float) - undvikningsvinkel [rad]
                is_wall       (bool)  - hindret klassificeras som vägg
                edge_angle_deg (float)- kantvinkel [grader]
                turn_sign     (float) - +1.0 = vänster, -1.0 = höger
        """
        if image is None: 
            return False, 0.0  , False , 0.0 , 1.0

        # Steg 1: Bird's Eye View 
        bev = self.to_bev(image)  

        # Steg 2: Förbearbeta bilden 
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)  
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) 

        # Steg 3: Ta bort golvet 
        floor_removed = self.remove_floor(blurred)   

        # Steg 4: Kantdetektion   
        edges = cv2.Canny(floor_removed, 30, 100)   

        # Steg 5: Konturanalys  
        contours, _  = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

        height, width = bev.shape[:2]
        roi_bottom = int(height * 0.7)
        best_contour = None       
        best_area = 0

        for contour in contours:   
            area = cv2.contourArea(contour)   
            if area < MIN_CONTOUR_AREA: 
                continue

            x , y,w,h = cv2.boundingRect(contour)   
            contour_center_y = y + h // 2   
            
            if contour_center_y < roi_bottom and area > best_area:
                best_area = area
                best_contour = contour            

        if best_contour is None:  
            return False, 0.0 , False , 0.0 , 1.0   # Inget hinder hittat
        
        # Steg 6: Beräkna fritt utrymme och bestäm svängriktning
        x , y,w,h = cv2.boundingRect(best_contour)
        left_space = x             
        right_space = width -(x+w)   
        rospy.loginfo(
            f"Foremål hittad: area={best_area:.0f}  "
            f"plats vänster={left_space}  höger={right_space}"
        )
        
        # Klassificera som vägg om utrymmet på båda sidor är för litet
        if left_space < WALL_THRESHOLD and right_space < WALL_THRESHOLD:
            rospy.loginfo_throttle(2.0, f"VÄGG från kamera bild  |  vänster={left_space} | höger={right_space} --> backar")
            return True, 0.0, True, 0.0 , 1.0
        
        # Välj sida med mest utrymme
        if left_space > right_space:
            edge_pixel = x
            side_text = "VÄNSTAR"
            turn_sign = 1.0
            rospy.loginfo("Mest plats på VÄNSTAR -> svänger vänstar")
        else:
            edge_pixel = x + w   # hindrets högra kant
            side_text = "HÖGER"
            turn_sign = -1.0     # negativ omega = höger
            rospy.loginfo("Mest plats på HÖGER -> svänger höger")
        
        # Konvertera kantpixelns position till undvikningsvinkel
        pixel_offset = (edge_pixel- width/2.0) /(width/2.0) * (CAMERA_HFOV /2) 
        angle_error = math.radians(max(-30.0, min(30.0, pixel_offset)))
        theta_avoid = self.normalize_angle(self.current_theta + angle_error)   
        edge_angle_deg = math.degrees(abs(angle_error)) 

        rospy.loginfo_throttle(
            3.0,
            f"Hinder | sida={side_text} | vänster={left_space}px | "
            f"höger={right_space}px | theta_avoid={math.degrees(theta_avoid):.1f}| "
            f"kant-vinkel={edge_angle_deg:.1f}grader"
        )

        return True, theta_avoid, False, edge_angle_deg , turn_sign
    
    #==============================
    # Huvudloop - tillståndsmaskin
    #==============================
    def run(self):
        """Kör tillståndsmaskinen i en loop med 20 Hz."""        
        rate = rospy.Rate(20)
        last_time = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()  
            dt  = max(0.001, min(0.1, now - last_time))
            last_time = now

            # Watchdog
            if self.check_state_timeout(now):
                rate.sleep()
                continue     

            # Omedelbar backning vid väggdetektering 
            if self.wall_detected and self.state != REVERSING:
                rospy.logwarn(f"wall_detected=True i [{self.state}] -> backar")
                self.obstacle_pub.publish(Bool(data=True))
                self.start_reversing()
                rate.sleep()
                continue
            
            # Väntar på initialvärde för målvinke
            if self.theta_original is None:
                rate.sleep()
                continue
            
            # IDLE: väntar på hinder
            if self.state ==IDLE: 
                if self.camera_active and  self.latest_image is not None :  
                    self.obstacle_pub.publish(Bool(data=True))   
                    self.stop()
                    self.scan_results= []  
                    self.scan_start = now  
                    self.state = SCANNING
                    self.state_entry_time = now
                    rospy.loginfo(f"IDLE->SCANNING | theta_orig={math.degrees(self.theta_original):.1f}")
            
            # SCANNING: samlar kameraresulta
            elif self.state== SCANNING:
                self.stop()
                if self.latest_image is not None: 
                    found, theta, is_wall, edge_deg, turn_sign = self.analys_image(self.latest_image)             
                    
                    if found and is_wall:
                        rospy.loginfo("SCANNING -> REVERSING (vägg via BEV-kamera)")
                        self.start_reversing()
                        rate.sleep(); continue  
                    if found:
                        safe_tof = min(self.tof_range if self.tof_initialized else TOF_THRESHOLD, MAX_TOF_FOR_DISTANCE)
                        self.scan_results.append((theta, edge_deg, safe_tof, turn_sign)) 
            
                ready = len(self.scan_results) >= SCAN_FRAMES     
                timeout = (now -self.scan_start) >= SCAN_TIMEOUT  

                if ready or (timeout and self.scan_results): 
                    # Beräkna median för robust undvikningsriktning    
                    thetas    = [r[0] for r in self.scan_results] 
                    edge_degs = [r[1] for r in self.scan_results] 
                    tof_vals  = [r[2] for r in self.scan_results] 
                    turn_signs = [r[3] for r in self.scan_results]
                    
                    self.theta_avoid = float(np.median(thetas))   
                    median_egde_deg = float(np.median(edge_degs)) 
                    median_tof = float(np.median(tof_vals))  
                    self.avoid_turn_sign = 1.0 if sum(turn_signs) >= 0 else -1.0
                    self.avoid_distance = self.compute_avoid_distance(median_tof, median_egde_deg)  
                    
                    self.theta_at_rotation_start = self.current_theta
                    self.distance_driven = 0.0
                    self.reset_pid()
                    self.state= ROTATING
                    self.state_entry_time = now
                    rospy.loginfo(f"SCANNING -> ROTATING |"
                    f"theta_avoid={math.degrees(self.theta_avoid):.1f} | "
                    f"sign={self.avoid_turn_sign:+.0f} | "
                    f"köravstånd={self.avoid_distance:.2f}m"
                    )
                elif timeout:   
                    rospy.loginfo("SCANNING -> SCAN_FALLBACK (kamera hittade inget, använder ToF)")
                    self.fallback_scan_start_theta = self.current_theta
                    self.fallback_scan_dir = -1.0  # Börja med att rotera åt höger
                    self.fallback_total_rotated = 0.0
                    self.fallback_prev_theta = self.current_theta
                    self.reset_pid()
                    self.state=SCAN_FALLBACK
                    self.state_entry_time = now

            # SCAN_FALLBACK: roterar med ToF för att hitta fri väg
            elif self.state ==SCAN_FALLBACK:
                delta = abs(self.normalize_angle(self.current_theta - self.fallback_prev_theta))
                delta= min(delta, math.radians(10))  # max 10° per tick

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
                elif self.fallback_rotated_per_dir >= FALLBACK_MAX_PER_DIR and self.fallback_scan_dir == -1.0:
                    # Inget åt höger - prova vänster istället
                    rospy.logwarn("SCAN_FALLBACK: ingen  år höger -> provar vänstar")
                    self.fallback_scan_dir = 1.0
                    self.fallback_scan_start_theta = self.current_theta
                    self.fallback_rotated_per_dir  = 0.0
                    self.stop()

                elif self.fallback_total_rotated >= FALLBACK_MAX_ROTATE:
                    # Ingen fri väg hittad - återgå till IDLE
                    rospy.logwarn("SCAN_FALLBACK: ingen FRI VÄGG -> IDLE ")
                    self.obstacle_pub.publish(Bool(data=False))
                    self.camera_active = False
                    self.state= IDLE
                    self.state_entry_time = now
                else: 
                    # Fortsätt rotera
                    self.twist_pub.publish(Twist2DStamped(v=0.0, omega=self.fallback_scan_dir * FALLBACK_ROTATE_SPEED))
                    rospy.loginfo_throttle(
                        3.0,
                        f"SCAN_FALLBACK | ToF={self.tof_range:.2f}m | "
                        f"riktning={'höger' if self.fallback_scan_dir < 0 else 'vänster'} | "
                        f"roterat={math.degrees(self.fallback_total_rotated):.1f}gradder"
                    )
            
            # FB_DRIVING: kör framåt i fallback-riktning
            elif self.state ==FB_DRIVING:
                self.fallback_drive_distance += AVOID_VELOCITY *dt
                if self.fallback_drive_distance>=FALLBACK_DRIVE:
                    self.stop()
                    self.reset_pid()
                    self.state=RETURNING
                    self.state_entry_time = now
                    rospy.loginfo(
                        f"FB_DRIVING -> RETURNING | "
                        f"kört={self.fallback_drive_distance:.2f}m"
                    )
                else:
                    if self.tof_range < TOF_WALL_DIST:
                        rospy.loginfo("FB_DRIVING:   VÄGG-> REVERSING")
                        self.start_reversing()
                    else:
                        omega = self.pid_steer(self.theta_avoid)
                        self.twist_pub.publish(Twist2DStamped(v=AVOID_VELOCITY, omega=omega))
                        rospy.loginfo_throttle(
                            2.0,
                            f"FB_DRIVING | {self.fallback_drive_distance:.2f}/{FALLBACK_DRIVE:.2f}m"
                        )

            # ROTATING: roterar mot undvikningsriktning
            elif self.state == ROTATING: 
                if self.rotate_to(self.theta_avoid, dt):
                    self.total_rotation_done = self.normalize_angle(self.current_theta - self.theta_at_rotation_start)
                    still_blocked = False
                    if self.latest_image is not None and self.tof_range < TOF_THRESHOLD:
                        found, _, _, _, _ = self.analys_image(self.latest_image)
                        still_blocked = found

                    if still_blocked:
                        # Justera undvikningsvinkeln ytterligare
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
            
            # AVOIDING: kör förbi hindret
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
                        self.distance_driven += AVOID_VELOCITY * dt
                        rospy.loginfo_throttle(2.0,
                            f"AVOIDING | "
                            f"{self.distance_driven:.2f}/{self.avoid_distance:.2f}m | "
                            f"omega={omega:.2f}"
                        )

            # REVERSING: backar från vägg      
            elif self.state == REVERSING:
                elapsed = now - self.reverse_start
                if elapsed < REVERSE_DURATION:
                    self.twist_pub.publish(Twist2DStamped(v=-REVERSE_SPEED, omega=0.0))
                    rospy.loginfo_throttle(
                        1.0, f"REVERSING | {elapsed:.1f}/{REVERSE_DURATION:.1f}s"
                    )
                else: 
                    self.stop()
                    self.reset_pid()
                    self.scan_results= []
                    self.scan_start= now
                    self.state= SCANNING
                    self.state_entry_time = now
                    rospy.loginfo("REVERSING -->>>> SCANNING")

            # RETURNING: roterar tillbaka till ursprungsriktning
            elif self.state == RETURNING:
                # Kontrollera om nytt hinder dyker upp på vägen tillbaka
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
        """Stoppar roboten säkert vid nedstängning."""
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