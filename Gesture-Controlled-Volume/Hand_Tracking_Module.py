import cv2 as cv
import mediapipe as mp
import time 
import math 

class handDetector():

    def __init__(self,mode = False, maxHands =2, detectionCon = 0.5, trackingCon = 0.5):
        
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        
        self.hands = self.mpHands.Hands(
        static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon,
        min_tracking_confidence=self.trackingCon )

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw = True):
       
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLMS, self.mpHands.HAND_CONNECTIONS)


        return img

    def findPosition(self,img, handNo = 0 ,draw = True ):

        lmList =[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id,lm in enumerate(myHand.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx,cy)
                lmList.append([id,cx,cy]) 

                if draw:
                    cv.circle(img,(cx,cy),3,(255,0,0),cv.FILLED)        
 
        return lmList
    
    def distance(self,lm_list, p1, p2):
        x1, y1 = lm_list[p1][1], lm_list[p1][2]
        x2, y2 = lm_list[p2][1], lm_list[p2][2]
        return math.hypot(x2 - x1, y2 - y1)

    def normalized_distance(self,lmList, p1, p2):
    #Distance between p1 and p2 as a ratio of hand size (camera-distance invariant).

        ref1 = self.distance(lmList, 0, 9)
        ref2 = self.distance(lmList, 5, 17)

        hand_size = max(ref1, ref2)
        normalized = self.distance(lmList, p1, p2) / hand_size    # wrist to middle knuckle = hand scale
        if hand_size == 0:
            return 0
        return normalized


