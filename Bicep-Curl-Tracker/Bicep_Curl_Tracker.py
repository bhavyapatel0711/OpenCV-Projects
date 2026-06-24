import cv2 as cv
import numpy as np
import Pose_Tracking_Module as ptm
import time

def main():
    # 1. Set webcam input and resolution
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    # 2. Create pose detector with requested 0.7 confidence
    detector = ptm.poseDetector(detectionCon=0.7, trackingCon=0.7)

    # 3. Initialize independent counters and direction trackers
    leftCount = 0.0
    rightCount = 0.0
    
    # Direction: 0 means going UP (curling towards shoulder), 1 means going DOWN
    leftDir = 0  
    rightDir = 0

    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Mirror the webcam for intuitive user interaction
        img = cv.flip(img, 1)

        # Detect the pose skeleton
        img = detector.findPose(img, draw=True)
        lmList = detector.findPosition(img, draw=True)

        if len(lmList) != 0:
            # Note: Because the image is flipped, the physical Left/Right sides 
            # of the person map to MediaPipe's Left/Right natively.
            
            # LEFT ARM (Shoulder: 11, Elbow: 13, Wrist: 15)
            angleL = detector.findAngle(img, 11, 13, 15, draw=True)
            
            # RIGHT ARM (Shoulder: 12, Elbow: 14, Wrist: 16)
            angleR = detector.findAngle(img, 12, 14, 16, draw=True)

            # Draw angle values explicitly near elbows for clarity
            cv.putText(img, f'{int(angleL)} deg', (lmList[13][1] - 50, lmList[13][2] + 40), 
                       cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            cv.putText(img, f'{int(angleR)} deg', (lmList[14][1] - 50, lmList[14][2] + 40), 
                       cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            # Map angles to percentage (100% = contracted, 0% = extended)
            perL = np.interp(angleL, (20, 160), (100, 0))
            perR = np.interp(angleR, (20, 160), (100, 0))
            
            # Map angles to pixel heights for the progress bars (150 to 650)
            barL = np.interp(angleL, (20, 160), (150, 650))
            barR = np.interp(angleR, (20, 160), (150, 650))

            # --- Left Arm Half-Rep Logic ---
            if perL == 100:
                if leftDir == 0:
                    leftCount += 0.5
                    leftDir = 1
            if perL == 0:
                if leftDir == 1:
                    leftCount += 0.5
                    leftDir = 0

            # --- Right Arm Half-Rep Logic ---
            if perR == 100:
                if rightDir == 0:
                    rightCount += 0.5
                    rightDir = 1
            if perR == 0:
                if rightDir == 1:
                    rightCount += 0.5
                    rightDir = 0

        # Main dashboard background and border
        cv.rectangle(img, (10, 10), (250, 115), (30, 30, 30), cv.FILLED)
        cv.rectangle(img, (10, 10), (250, 115), (255, 255, 255), 2)

        # Current stages
        l_stage = "UP" if leftDir == 0 else "DOWN"
        r_stage = "UP" if rightDir == 0 else "DOWN"

        # Dashboard Text Layout
        cv.putText(img, f'RIGHT  : {int(leftCount)} ', (20, 50), 
                   cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
        cv.putText(img, f'LEFT : {int(rightCount)} ', (20, 95), 
                   cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

        # 6. FPS Display (Top-Right)
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        
        # FPS Background and border
        cv.rectangle(img, (1080, 20), (1250, 70), (30, 30, 30), cv.FILLED)
        cv.rectangle(img, (1080, 20), (1250, 70), (0, 255, 255), 2)
        cv.putText(img, f'FPS: {int(fps)}', (1100, 55), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Display the final image frame
        cv.imshow('Professional Bicep Curl Counter', img)
        
        # 7. Press Q to Quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()