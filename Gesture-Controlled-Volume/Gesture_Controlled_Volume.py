import cv2 as cv
import time
import numpy as np
import Hand_Tracking_Module as htm
import math as ma
import pycaw 
from pycaw.pycaw import AudioUtilities

wCam, hCam = 1280,720  
cap = cv.VideoCapture(0)
cap.set(3,wCam)     
cap.set(4,hCam)
pTime = 0
 
detector = htm.handDetector(detectionCon= 0.7)

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
smoothVolPer = 0
smoothVolBar = 398

minLength = float('inf')
maxLength = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    # Glassmorphism Panels
    overlay = img.copy()

    cv.rectangle(overlay, (20, 90), (150, 460), (35, 35, 35), -1)
    cv.rectangle(overlay, (20, 20), (170, 70), (35, 35, 35), -1)

    alpha = 0.45
    img = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    
    # Hand Detection
    if len(lmList) != 0:

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = detector.normalized_distance(lmList, 4, 8)

        # Convert Hand Distance to Volume
        vol = np.interp(length,[0.16, 1.4], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)
        
        volBar = np.interp(length, [0.16, 1.4],[398, 152])
        volPer = np.interp(length,[0.16, 1.4],[0, 100])

        # Smooth Animation
        smoothVolBar = 0.8 * smoothVolBar + 0.2 * volBar
        smoothVolPer = 0.5 * smoothVolPer + 0.5 * volPer

        lineColor = (0,80,255)
        # Draw Fingers
        cv.circle(img, (x1, y1), 15, lineColor, cv.FILLED)
        cv.circle(img, (x2, y2), 15, lineColor, cv.FILLED)

        cv.line(img,(x1, y1),(x2, y2),lineColor,5)

        cv.circle(img,(cx, cy), 12,lineColor,cv.FILLED)

        if length < 0.16:
            cv.circle(img,(cx, cy),18,(0, 255, 0),cv.FILLED)
    
        # Volume Percentage Box
        cv.rectangle(img,(25, 415),(145, 455),(45, 45, 45),-1)
        cv.putText(img,f'{min(100, round(smoothVolPer))}%',(45, 445),cv.FONT_HERSHEY_DUPLEX,1,(255, 255, 255),2)

    # Volume Bar
    cv.rectangle(img, (50, 150), (90, 400), (255, 255, 255), 2)

    for i in range(int(smoothVolBar), 400):

        ratio = (400 - i) / 250
        color = ( 255,int(255 * ratio),int(100 * ratio))
        cv.line(img,(52, i),(88, i),color,1)

    # Volume Status
    if smoothVolPer < 30:
        status = "LOW"
        statusColor = (255, 150, 0)

    elif smoothVolPer < 70:
        status = "MED"
        statusColor = (0, 255, 255)

    else:
        status = "HIGH"
        statusColor = (0, 255, 0)

    cv.putText(img,status, (45, 125),cv.FONT_HERSHEY_DUPLEX,0.8,statusColor,2)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img,f'FPS : {int(fps)}',(35, 55), cv.FONT_HERSHEY_DUPLEX,0.9,(0, 255, 255),2)
    
    cv.imshow('Gesture Volume Controller', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()   
cv.destroyAllWindows() 