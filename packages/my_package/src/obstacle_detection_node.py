import os
import cv2       #  används för att bearbeta kamerabilder
import math
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import Range, CompressedImage
from std_msgs.msg import Bool
from cv_bridge import CvBridge


TOF_THRESHOLD   = 0.5    # meter — triggar kameran om föremål är närmare än detta
AVOID_OMEGA     = 2.0    #  hur hårt roboten svänger för att undvika (rad/s )
AVOID_VELOCITY  = 0.2    #  hastighet under undvikande
AVOID_DURATION  = 1.5    # hur länge roboten svänger (sek)

 
class ObstacleDetectionNode(DTROS):

    def __init__(self, node_name):
        super(ObstacleDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']

        self.twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.tof_topic = f"/{self.vehicle_name}/front_center_tof_driver_node/range"
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        
        # Topic som skickar signal till twist_control_node
        self.obstacle_topic = f"/{self.vehicle_name}/obstacle_detection_node/obstacle_detected"


        self.tof_range = float ('inf')     # Senaste avståndet från ToF
        self.camera_active = False         # True när kameran analyserar
        self.obstacle_detected  = False
        self.last_image = None          # senaste kameranild
        

        
        self.obstacle_pub = rospy.Publisher(self.obstacle_topic, Bool,  queue_size=1)

    def callback_tof(self, msg):   #  Körs varje gång ToF-sensorn skickar ett nytt avstånd.

        self.tof_range = msg.range      # Sparar senaste avståndet
        if msg < TOF_THRESHOLD:         # Kolla om avståndet är mindre än tröskelvärdet
            if not self.camera_active:           # Om kameran inte redan är aktiv
                rospy.loginfo(f"TOF: Formål pa {msg.range:.2f}m --> aktivera kamera")
                self.camera_active = True
        else:
            if self.camera_active:             # Om avståndet är större än tröskeln
                rospy.loginfo(f"ToF: Vagen klar ({msg.range:.2f}m) — kamera inaktiv")
                self.camera_active = False     # Inget hinder nära --> stäng av kamera-analys
                self.obstacle_detected = False

    def callback_camera(self, msg):     #   Körs varje gång kameran skickar en ny bild.

        if not self.camera_active:      # Hoppa över om kameran inte är aktiv
            return
        try: 

            # Gör om den komprimerade bilden till OpenCV-format
            np_arr = np.frombuffer(msg.date, np.uint8)   # np.frombuffer = konverterar dess till en numpy-arry  (np.unit8= varje värde tolkas som ett tal 0-225)

            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)   # cv2.imdecode() tar numpy-arry och dekoder den till en riktig bild / cv2. IMREAD_COLOR = säger att bilden ska läsas som en färgbild
            self.last_image = image
            
            obstacle_found = self.analys_image(image)         # Analyserar bilden


            # Om hinder hittas och det inte redan var upptäckt
            if obstacle_found and not self.obstacle_detected:
                rospy.loginfo("Kamera: Foremal bekräftat — undviker!")
                self.obstacle_detected = True

                    # Skickar signal att hinder finns
                self.obstacle_pub.publish(Bool(data=True))
                self.avoid_obstacle(self)
        except Exception as e:    # Om något går fel--> krascha inte, skriv ut ett felmeddelande istället
            rospy.logwarn(f"Kamerafel: {e}")


    def analys_image (self, image):   # analysera  komerabild för att bekräfta föremål. 
        if image in None: 
            return False, 0
        height, width = image.shape[:2]    # Hämtar bildens höjd och bredd 

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
        
        #  Kolla om det finns en stor kontur i mitten av bilden
        center_x = width // 2
        center_y = height // 2    

        for contour in contours: 
            area = cv2.contourArea(contour)   # Storlek på objektet

            if area < 1000: # Ignorera små konturer (brus)
                continue

            # Hämtar en rektangel runt konturen
            x, y,w,h = cv2.boundingRect(contour)    # Lätt att hitta mitten av objektet

            contour_center_x = x + w // 2     # objekts mittpunkten
            contour_center_y = y + h // 2
            
            # Om konturen är i mitten +,-30% av bilden --> föremål bekräftat
            # Vi bryr oss bara om objekt i mitten (t.ex. hinder rakt fram)
            if (abs(contour_center_x - center_x) < width  * 0.3     # 0.3 Tillåter lite felmarginal
                and abs(contour_center_y - center_y) < height * 0.3):
                rospy.loginfo(
                    f"Kontur hittad: area={area:.0f}  "
                    f"pos=({contour_center_x}, {contour_center_y})"
                )
                return True 
            
        return False
        
    def avoid_obstacle(slef):   # undvik förmål genom att sbänga 



        pass
            

    pass