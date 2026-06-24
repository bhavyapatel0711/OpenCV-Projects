import cv2 as cv
import mediapipe as mp
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

        self.handedness = []

        if self.results.multi_handedness:
            for hand in self.results.multi_handedness:
                self.handedness.append(
                    hand.classification[0].label
                )
        
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLMS, self.mpHands.HAND_CONNECTIONS)


        return img

    # def findPosition(self,img, handNo = 0 ,draw = True):

    #     lmList =[]
    #     if self.results.multi_hand_landmarks:
    #         myHand = self.results.multi_hand_landmarks[handNo]

    #         for id,lm in enumerate(myHand.landmark):
    #             #print(id,lm)
    #             h,w,c = img.shape
    #             cx,cy = int(lm.x * w), int(lm.y * h)
    #             #print(id, cx,cy)
    #             lmList.append([id,cx,cy]) 

    #             if draw:
    #                 cv.circle(img,(cx,cy),3,(255,0,0),cv.FILLED)        
 
    #     if self.results.multi_handedness:
    #         label = self.handedness[handNo]
    #     else:
    #         label = None

    #     return lmList, label

    def findPosition(self, img, draw = True):

        allHands = []

        if self.results.multi_hand_landmarks:

            h, w, c = img.shape

            for handNo, handLms in enumerate(
                self.results.multi_hand_landmarks
            ):

                lmList = []

                for id, lm in enumerate(handLms.landmark):

                    cx = int(lm.x * w)
                    cy = int(lm.y * h)

                    lmList.append([id, cx, cy])

                    if draw:
                        cv.circle(img, (cx, cy), 3, (255, 0, 0), cv.FILLED)

                label = (
                    self.handedness[handNo]
                    if hasattr(self, "handedness")
                    else None
                )

                allHands.append({
                    "lmList": lmList,
                    "label": label
                })

        return allHands
    

