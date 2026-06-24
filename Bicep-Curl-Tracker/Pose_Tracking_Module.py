import cv2 as cv
import mediapipe as mp
import time
import math

class poseDetector():

    def __init__(self, mode=False, model_complexity=1, smooth_landmarks=True, detectionCon=0.5, trackingCon=0.5):
        
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode, 
            model_complexity=self.model_complexity, 
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon 
        )
        self.mpDraw = mp.solutions.drawing_utils
        
        # Initialize results and lmList to prevent errors if called out of order
        self.results = None
        self.lmList = []

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results and self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        self.lmList = [] # Reset the list every frame
        
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Appending id, x, y, and visibility
                self.lmList.append([id, cx, cy, lm.visibility]) 

                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)        
 
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Calculates the angle between three landmarks. p2 is the vertex.
        """
        if len(self.lmList) != 0:
            # Get the coordinates
            x1, y1 = self.lmList[p1][1:3]
            x2, y2 = self.lmList[p2][1:3]
            x3, y3 = self.lmList[p3][1:3]

            # Calculate the angle using atan2
            angle = math.degrees(
                math.atan2(y3 - y2, x3 - x2) -
                math.atan2(y1 - y2, x1 - x2)
            )

            if angle < 0:
                angle += 360

            if angle > 180:
                angle = 360 - angle

            # Draw the lines and angle on the image
            if draw:
                cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
                
                cv.circle(img, (x1, y1), 10, (0, 0, 255), cv.FILLED)
                cv.circle(img, (x1, y1), 15, (0, 0, 255), 2)
                cv.circle(img, (x2, y2), 10, (0, 0, 255), cv.FILLED)
                cv.circle(img, (x2, y2), 15, (0, 0, 255), 2)
                cv.circle(img, (x3, y3), 10, (0, 0, 255), cv.FILLED)
                cv.circle(img, (x3, y3), 15, (0, 0, 255), 2)
                
            return angle
        return 0

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        # Optional: Mirror the webcam
        img = cv.flip(img, 1)

        img = detector.findPose(img, draw=True) # Set to True to see all skeletal connections
        lmList = detector.findPosition(img, draw=True)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, 'Fps: ' + str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        
        cv.imshow('Video', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break      

    cap.release()   
    cv.destroyAllWindows() 

if __name__ == '__main__':
    main()