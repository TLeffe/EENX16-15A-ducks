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

try:
    from ultralytics import YOLO     # Försök importera YOLO från ultralytics
    YOLO_AVAILABLE = True           # Sätt flagga till True om import lyckades
except ImportError:
    YOLO_AVAILABLE = False           # Om import misslyckas finns inte YOLO tillgängligt
#------Avståndsgränser
TOF_THRESHOLD = 0.40           # meter - triggar kameran om föremål är närmare än detta
TOF_WALL_DIST = 0.17   # meter - under detta = vägg, backa direkt utan kamera
TOF_CLEAR_DIST  = 0.50   # meter - används i ToF-fallback scan
FALLBACK_DRIVE  = 0.25   # meter - hur långt vi kör i fallback-läge
MAX_TOF_FOR_DISTANCE = 0.80   # m - skyddar Pythagoras mot 100m-värden 
MIN_TOF_VALID        = 0.02   # m - under detta = brus, ignorera

#-----Hastigheter-------
AVOID_VELOCITY = 0.15            #  hastighet under undvikande
REVERSE_SPEED = 0.15           #  # m/s bakåt
FALLBACK_ROTATE_SPEED = 0.7   # rad/s konstant omega vid fallback-scan (enklare än PID)
REVERSE_DISTANCE = 0.20 # m 
REVERSE_DURATION = REVERSE_DISTANCE / REVERSE_SPEED     # Tid för att backa 20 cm

#---- camera------
CAMERA_HFOV = 160.0             # grader - kamerans horisontella synfält på Duckiebot
IMG_W = 640
IMG_H = 480
MAX_AVOID_ANGLE = math.radians(25)       # Max vinkel för undvikande i radianer

# -- YOLO
YOLO_CONF = 0.45   # confidence-tröskel -höj om för många falska positiver
YOLO_IMGSZ  = 320    # inferensstorlek - 320 är snabbast på Jetson Nano
SCAN_FRAMES   = 4      # antal frames att samla i SCANNING
SCAN_TIMEOUT  = 1.5    # s - max tid i SCANNING

# OBSTACLE_CLASSES: lista med klassnamn att reagera på.
# None = reagera på ALLA klasser (bra för test).
# Exempel för Duckietown: ["duckie", "duckiebot", "cone"]
# Exempel för generell COCO-modell: ["person", "chair", "bottle"]
OBSTACLE_CLASSES = None

WALL_COVER_FRAC = 0.75 # Vägg-tröskel: om hindret täcker > WALL_COVER_FRAC av bildbredden -> vägg

ROI_BOTTOM_FRAC = 0.80 # ROI bara övre ROI_BOTTOM_FRAC av bilden räknas (hindret ska vara framför)

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
STATE_TIMEOUT = 6.0

#  Kamera-intrinsics (från er kalibreringsfil) 
# Används för att rätta linsförvrängning innan BEV-transformation.
CAMERA_MATRIX = np.array([
    [309.34488578183976, 0.0, 318.97431157284797],
    [0.0, 324.3079313761394, 249.74718090589798],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DIST_COEFFS = np.array([
    -0.26850619979326934, 0.14095182385200886,
    0.029986662850901284,0.000792436879591894,
    -0.03422901917827191
], dtype=np.float64)

# ── YOLO-modellsökvägar (prioritetsordning) ────────────────────────────────────
YOLO_MODEL_PATHS = [
    "/data/weights/yolov8n.pt",                              # generell fallback
]

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
        self.tof_initialized = False
        self.camera_active = False         # True när kameran analyserar
        self.state = IDLE            # State machine — börjar i IDLE
        self.wall_detected = False
        self.state_entry_time = 0.0

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
        
        self.fallback_scan_dir = -1.0   # -1 = höger, +1 = vänster
        self.fallback_scan_start_theta = 0.0  # vinkel när fallback-scan startade
        self.fallback_drive_distance   = 0.0
        self.fallback_total_rotated = 0.0
        self.fallback_prev_theta = 0.0
        self.fallback_rotated_per_dir = 0.0

        self._integral = 0.0              #   PID-sate
        self._prev_error = 0.0

        self._lock = threading.Lock()
        self._latest_img = None   # råbild (undistortad)
        # img_result: (found, theta_avoid, is_wall, edge_angle_deg, turn_sign, timestamp)
        self._img_result  = (False, 0.0, False, 0.0, 1.0, 0.0)
        self._img_running = True

        #  Undistortion-karta (beräknas EN gång, snabb sedan) 
        # cv2.initUndistortRectifyMap ger pixelkartor -> remap är ungefär 3 ms
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            CAMERA_MATRIX, DIST_COEFFS, None, CAMERA_MATRIX, (IMG_W, IMG_H), cv2.CV_16SC2)
        rospy.loginfo("Undistortion-karta beräknad")
        
        # YOLO
        self.model = self.load_yolo()
        self.class_names  = self.model.names if self.model else {}

        # Starta bildtråd
        threading.Thread(target=self.yolo_loop, daemon=True).start()
        
        self.obstacle_pub = rospy.Publisher(self.obstacle_topic, Bool, queue_size=1)    #  Publisher som skickar sant/falskt om hinder finns
        self.twist_pub = rospy.Publisher(self.twist_topic, Twist2DStamped, queue_size=1)  # Publisher som skickar hastighetskommandon till roboten
        
        rospy.Subscriber(self.tof_topic, Range, self.callback_tof)      # Lyssnar på ToF-sensorn och kör callback_tof vid nytt meddelande
        rospy.Subscriber(self.camera_topic, CompressedImage, self.callback_camera)   # Lyssnar på kameran och kör callback_camera vid ny bild
        rospy.Subscriber(self.desired_theta_topic, Float32, self.callback_desired_theta)
        rospy.Subscriber(self.current_theta_topic, Float32,self.callback_current_theta)

        rospy.loginfo(f"ObstacleDetectionNode startad")

    def load_yolo(self):
        if not YOLO_AVAILABLE:
            rospy.logwarn("ultralytics inte installerat -------  kör utan YOLO")  # Logga varning
            return None
        for path in YOLO_MODEL_PATHS:      # Gå igenom alla möjliga modellvägar
            #if os.path.exists(path):       # Om filen finns
            try:
                m = YOLO(path)         # Ladda modellen från filen
                m(np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8),   # Gör en testinferens för att värma upp modellen
                    verbose=False, imgsz=YOLO_IMGSZ)
                rospy.loginfo(f"YOLO laddad: {path}")
                rospy.loginfo(f"Klasser: {list(m.names.values())}")
                return m
            except Exception as e:
                rospy.logwarn(f"Kunde inte ladda {path}: {e}")
        return None


    def yolo_loop(self):
        rate = rospy.Rate(12)
        while not rospy.is_shutdown() and self._img_running:    # Kör tills ROS stängs ned eller flaggan sätts till False
            with self._lock:                                     # Lås för trådsäker åtkomst
                img = self._latest_img                            # Hämta senaste bild

            if img is not None and self.camera_active and self.model is not None:
                result = self.analyse(img)               # Kör bildanalysen
                with self._lock:                          # Lås igen för att skriva resultatet säkert
                    self._img_result = result
            rate.sleep()   

    def get_result(self):                           # Hjälpfunktion för att hämta senaste bildresultatet
        with self._lock:                            # Lås för trådsäker läsning
            return self._img_result                 # Returnera senaste sparade resultat

    def analyse(self, image):                        # Bildanalys med YOLO

        if image is None or self.model is None:
            return False, 0.0, False, 0.0, 1.0, rospy.Time.now().to_sec()

        height, width = image.shape[:2]              # Hämta bildens höjd och bredd
        roi_bottom = int(height * ROI_BOTTOM_FRAC)   # Beräkna nedre ROI-gräns
        timestamp  = rospy.Time.now().to_sec()

        try:     # Kör YOLO-inferens på bilden med vald confidence och inferensstorlek.
            results = self.model(image, verbose=False, conf=YOLO_CONF, imgsz=YOLO_IMGSZ)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"YOLO-inferens fel: {e}")
            return False, 0.0, False, 0.0, 1.0, timestamp

        best_box  = None       # Här lagras bästa bounding box.
        best_area = 0.0        # Här lagras största area hittills.       

        for box in results[0].boxes:    # Iterera över alla upptäckta bounding boxes i första resultatet.
            if OBSTACLE_CLASSES is not None:    # Om klassfiltrering används:
                cls_id   = int(box.cls[0])      # Hämta klass-ID för boxen
                cls_name = self.class_names.get(cls_id, "")    # Översätt klass-ID till klassnamn.
                if cls_name not in OBSTACLE_CLASSES:         # Hoppa över objekt som inte är intressanta hinder.
                    continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())    # Hämta boxens koordinater
            cy   = (y1 + y2) // 2        # Beräkna center-y för boxen.
            area = (x2 - x1) * (y2 - y1)    # Beräkna boxens area.

            # Bara objekt i framåt-ROI
            if cy < roi_bottom and area > best_area:     # Om objektet ligger inom ROI och är större än tidigare bästa
                best_area = area
                best_box  = (x1, y1, x2, y2)          # Spara denna box som bästa kandidat.

        if best_box is None:                        # Om inget lämpligt objekt hittades
            return False, 0.0, False, 0.0, 1.0, timestamp

        x1, y1, x2, y2 = best_box          # Packa upp bästa boxen.
        left_space  = x1                    # Fritt utrymme till vänster om hindret
        right_space = width - x2            # Fritt utrymme till höger om hindret.
        box_width   = x2 - x1             # Hindrets bredd i pixlar

        rospy.loginfo(
            f"YOLO: area={best_area:.0f} "
            f"Vänster={left_space} Höger={right_space} "
            f"box_w={box_width} img_w={width}"
        )

        if box_width > width * WALL_COVER_FRAC:          # Om hindret täcker för stor del av bilden
            rospy.loginfo_throttle(2.0,
                f"VÄGG (YOLO): box_w={box_width} > {width * WALL_COVER_FRAC:.0f}px")
            return True, 0.0, True, 0.0, 1.0, timestamp

    
        if left_space > right_space:          # Om det finns mer plats på vänster sida
            edge_pixel = x1    # vänster kant på hindret
            turn_sign  = 1.0   # sväng vänster
            rospy.loginfo("YOLO: mest plats VÄNSTER")
        else:
            edge_pixel = x2    # höger kant på hindret
            turn_sign  = -1.0  # sväng höger
            rospy.loginfo("YOLO: mest plats HÖGER")

        pixel_offset_deg = (edge_pixel - width / 2.0) / (width / 2.0) * (CAMERA_HFOV / 2.0)          # Översätt pixelpositionen till vinkel i grader relativt bildcentrum.
        angle_rad = math.radians(pixel_offset_deg)      # Omvandla vinkeln från grader till radianer.
        angle_rad = max(-MAX_AVOID_ANGLE, min(MAX_AVOID_ANGLE, angle_rad))      # Begränsa vinkeln till max 25 grader
        theta_avoid  = self.normalize_angle(self.current_theta + angle_rad)       # Målvinkel = nuvarande vinkel + undanmanövervinkeln
        edge_angle_deg   = math.degrees(abs(angle_rad))      # Sparar absolutvinkeln i grader för senare avståndsberäkning

        rospy.loginfo_throttle(3.0,
            f"YOLO: theta_avoid={math.degrees(theta_avoid):.1f} grader"
            f"kant={edge_angle_deg:.1f} grader sign={turn_sign:+.0f}")

        return True, theta_avoid, False, edge_angle_deg, turn_sign, timestamp

    def callback_desired_theta (self, msg):
        if self.state == IDLE:               # Behåll originalvinkeln fryst under undvikande
            if self.theta_original is None:
                rospy.loginfo(f"theta_original initierad: {math.degrees(msg.data):.1f}grader")    # Logga första gången theta_original sätts
            self.theta_original= msg.data     # Spara önskad riktning som "ursprunglig" riktning

    def callback_current_theta(self, msg):   # Tar emot robotens aktuella vinkel för PID-reglering
        self.current_theta = msg.data
        
    def callback_tof(self, msg):   #  Körs varje gång ToF-sensorn skickar ett nytt avstånd.
        if msg.range < MIN_TOF_VALID or msg.range > 5.0:    # Ignorera orimliga avstånd
            return
        
        self.tof_range = msg.range      # Sparar senaste avståndet
        self.tof_initialized = True
        if msg.range < TOF_WALL_DIST and self.state != REVERSING:    # Om något är väldigt nära och vi inte redan backar
            self.wall_detected = True
            rospy.loginfo(f"TOF: Formål på {msg.range:.2f}m ----> aktivera kamera")
            return
        
        if msg.range < TOF_THRESHOLD:         # Om hinder är närmare än tröskeln
            if not self.camera_active:
                self.camera_active = True      # Aktivera kamerabearbetningen
        else:        # Om ToF inte längre ser något nära
            if self.state == IDLE:         # Om avståndet är större än tröskeln
                rospy.loginfo(f"ToF: Vägen klar ({msg.range:.2f}m) ------> kamera inaktiv")
                self.camera_active = False     # Inget hinder nära --> stäng av kamera-analys
                self.obstacle_pub.publish(Bool(data=False))  

    def callback_camera(self, msg):     #   Körs varje gång kameran skickar en ny bild.
        if not self.camera_active and self.state == IDLE:      # Hoppa över om kameran inte är aktiv
            return
        try: 
            np_arr = np.frombuffer(msg.data, np.uint8)   # np.frombuffer = konverterar dess till en numpy-arry  (np.unit8= varje värde tolkas som ett tal 0-225)
            raw = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)   # cv2.imdecode() tar numpy-arry och dekoder den till en riktig bild / cv2. IMREAD_COLOR = säger att bilden ska läsas som en färgbild
            if raw is not None:
                undistorted = cv2.remap(raw, self._map1, self._map2, cv2.INTER_LINEAR)       # Korrigerar linsdistorsion
                with self._lock:
                    self._latest_img = undistorted    # Sparar senaste undistorterande bild trådsäkert.
                            
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

    def pid_steer(self, target, dt):
        if target is None or dt <= 0:
            return 0.0
        error = self.angle_error_to(target)
        derivative = (error - self._prev_error) / dt      # Derivatterm = förändring i fel per tidssteg
        self._prev_error = error

        self._integral  = max(-3.0, min(3.0, self._integral + error * dt))     # Uppdatera integraldel men begränsa den för att undvika windup
        omega = KP_THETA * error + KI_THETA * self._integral + KD_THETA * derivative
        omega = max(-OMEGA_MAX, min(OMEGA_MAX, omega))
        return omega         # Returnera PID-signal begränsad till max vinkelhastighet

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
    
    def start_reversing(self):     # Gemensam funktion för att starta backning
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
        result = max(MIN_AVOID_DISTANCE, min(MAX_AVOID_DISTANCE, c))   # Begränsa resulterande körsträcka mellan min och max
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
            self.state = IDLE
            self.state_entry_time = now
            return True
        return False
    
    def run(self):
        rate = rospy.Rate(20)
        last_time = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()    # Hämtar nuvarande ROS-tid i sekunder
            dt  = max(0.001, min(0.1, now - last_time))      # Beräkna tidssteg, begränsat mellan 0.001 och 0.1 s
            last_time = now
            
            found, theta_img, is_wall, edge_deg, turn_sign, img_ts = self.get_result()    # Hämta senaste YOLO-resultat från bildtråden.
            result_age = max(0.0, now - img_ts)     # Hur gammalt bildresultatet är
            if result_age > 0.5:    # Ignorera för gamla bildresultat
                found = False            
            
            if self.check_state_timeout(now):    # Om timeout inträffade, hoppa till nästa loopvarv
                rate.sleep()
                continue

            if self.wall_detected and self.state != REVERSING:
                rospy.logwarn(f"wall_detected=True i [{self.state}] --->>> backar")
                self.obstacle_pub.publish(Bool(data=True))
                self.start_reversing()
                rate.sleep()
                continue

            if self.theta_original is None:
                rate.sleep()         # Vänta tills theta_original har initierats
                continue   

            if self.state ==IDLE:   # IDLE--->> väntar på hinder
                if self.camera_active and self._latest_img is not None:  # Om kamera är aktiv och bild finns
                    self.obstacle_pub.publish(Bool(data=True))    # Publicera att hinder finns
                    self.stop()
                    self.scan_results= []     # Töm gamla scanresultat
                    self.scan_start = now     # Spara scan-starttid
                    self.state = SCANNING
                    self.state_entry_time = now     # Spara tillståndstid
                    rospy.loginfo(f"IDLE--->SCANNING | theta_orig={math.degrees(self.theta_original):.1f}")
            
            elif self.state== SCANNING:
                self.stop()
                if found and is_wall:
                    rospy.loginfo("SCANNING --> REVERSING (vägg via BEV-kamera)")
                    self.start_reversing()
                    rate.sleep()
                    continue     # Vänta och hoppa till nästa loop
                if found:
                    safe_tof = min(self.tof_range if self.tof_initialized else TOF_THRESHOLD, MAX_TOF_FOR_DISTANCE)
                    self.scan_results.append((theta_img, edge_deg, safe_tof, turn_sign))     # Spara undvikande vinkel
        
                ready = len(self.scan_results) >= SCAN_FRAMES     # Kolla om tillräckligt många bilder samlats
                timeout = (now -self.scan_start) >= SCAN_TIMEOUT   # Kolla om tiden tagit slut

                if ready or (timeout and self.scan_results):       # Om redo eller timeout med resultat
                    thetas    = [r[0] for r in self.scan_results]  # Plockar ut ALLA första värden (theta-vinklar)
                    edges = [r[1] for r in self.scan_results]  #  Plockar ut ALLA andra värden (kantvinklar i grader)
                    tofs  = [r[2] for r in self.scan_results]  # Plockar ut ALLA tredje värden (ToF-avstånd)
                    signs = [r[3] for r in self.scan_results]    # Samla alla svängtecken
                    
                    self.theta_avoid = float(np.median(thetas))      # Ta median av vinklarna
                    self.avoid_turn_sign = 1.0 if sum(signs) >= 0 else -1.0       # Välj övergripande svängriktning via majoritet.
                    self.avoid_distance  = self.compute_avoid_distance(float(np.median(tofs)), float(np.median(edges)))       # Beräkna undanmanöversträckan med medianvärden.
                    self.theta_at_rotation_start = self.current_theta         # Spara startvinkel innan rotation
                    self.distance_driven = 0.0
                    self.reset_pid()
                    self.state = ROTATING
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
                    self.fallback_rotated_per_dir = 0.0
                    self.fallback_prev_theta = self.current_theta
                    self.reset_pid()
                    self.state=SCAN_FALLBACK
                    self.state_entry_time = now

            elif self.state ==SCAN_FALLBACK:
                delta = min(abs(self.normalize_angle(self.current_theta - self.fallback_prev_theta)), math.radians(10))  #Beräkna ungefär hur mycket roboten roterat sedan senaste varv, max 10 grader  per steg
                self.fallback_total_rotated += delta     # Uppdatera total fallback-rotation
                self.fallback_rotated_per_dir += delta    # Uppdatera rotation i nuvarande riktning
                self.fallback_prev_theta     = self.current_theta
            
                if self.tof_range > TOF_CLEAR_DIST:   # Om ToF nu visar fri väg
                    self.theta_avoid = self.current_theta
                    self.fallback_drive_distance = 0.0
                    self.reset_pid()
                    self.state=FB_DRIVING
                    self.state_entry_time = now
                    rospy.loginfo(
                        f"SCAN_FALLBACK -->FB_DRIVING  |"
                        F" Riktning 0 {math.degrees(self.theta_avoid):.1f}"
                    )
                elif (self.fallback_rotated_per_dir >= FALLBACK_MAX_PER_DIR and self.fallback_scan_dir == -1.0):   #  prova vänster
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
                elif self.tof_range < TOF_WALL_DIST:
                    rospy.loginfo("FB_DRIVING:   VÄGG---->> REVERSING")
                    self.start_reversing()
                else:
                    self.twist_pub.publish(Twist2DStamped(v=AVOID_VELOCITY, omega=self.pid_steer(self.theta_avoid, dt)))
                    rospy.loginfo_throttle(
                        2.0,
                        f"FB_DRIVING | {self.fallback_drive_distance:.2f}/{FALLBACK_DRIVE:.2f}m"
                    )

            elif self.state == ROTATING:        # Om tillstånd är ROTATING
                if self.rotate_to(self.theta_avoid, dt):
                    self.total_rotation_done = self.normalize_angle(self.current_theta - self.theta_at_rotation_start)   # Räkna hur mycket bilen faktiskt roterat
                    still_blocked = found and (result_age < 0.3) and self.tof_range < TOF_THRESHOLD     # Kolla om hindret fortfarande verkar vara kvar framför bilen
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
                        self.state_entry_time = now
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
                if (now - self.avoid_start_time) > MAX_AVOID_TIME:      # Om undanmanövern tagit för lång tid
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
               
                elif self.tof_range < TOF_WALL_DIST:
                    rospy.loginfo("AVOIDING: vägg → REVERSING")
                    self.start_reversing()
                else:
                    omega = self.pid_steer(self.theta_avoid,dt)
                    self.twist_pub.publish(Twist2DStamped(v=AVOID_VELOCITY, omega=omega))
                    self.distance_driven += AVOID_VELOCITY * dt # Uppdatera hur långt vi kört (tid * hastighet)
                    rospy.loginfo_throttle(2.0,
                        f"AVOIDING | "
                        f"{self.distance_driven:.2f}/{self.avoid_distance:.2f}m | "
                        f"omega={omega:.2f}"
                    )
            elif self.state == REVERSING:
                elapsed = now - self.reverse_start      # Hur länge vi har backat
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
                if self.tof_range < TOF_THRESHOLD and found and result_age < 0.3:    # Om nytt hinder dyker upp på vägen tillbaka
                    self.scan_results = []
                    self.scan_start = now
                    self.stop()
                    self.state = SCANNING
                    self.state_entry_time = now
                    rospy.loginfo("RETURNING -> SCANNING (nytt hinder)")
                    rate.sleep()
                    continue
                    
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
        self._img_running = False
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
