import cv2 as cv
import time
from Hand_Tracking_Module import handDetector
from Gesture_Detection_Module import GestureDetector

def main():

    cap = cv.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = handDetector(mode=False, maxHands=2, detectionCon=0.7, trackingCon=0.7)
    gestureDetector = GestureDetector(history_length=5, confidence_threshold=0.6)
    pTime = 0

    while cap.isOpened():

        success, img = cap.read()
        img = cv.flip(img, 1)

        # Detect hands
        img = detector.findHands(img)

        hands = detector.findPosition(img)

        multi_hand_landmarks = [
            hand["lmList"]
            for hand in hands
        ]

        # --------- Gesture Detection ---------
        gesture = gestureDetector.process_frame(
            multi_hand_landmarks
        )

        # ---------- HUD PANEL ----------
        cv.rectangle(img, (10, 10), (620, 210), (30, 30, 30), cv.FILLED )

        cv.rectangle(img, (10, 10), (620, 210), (80, 80, 80), 2)

        # Gesture Title
        cv.putText(img, "Gesture:",(25, 45), cv.FONT_HERSHEY_COMPLEX, 0.7, (203 , 192, 255) , 2)

        # Detected Gesture
        cv.putText(img, f"{gesture}", (25, 90),cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 0), 3)

        # FPS   
        cTime = time.time()
        fps = 1 / (cTime - pTime + 1e-6)
        pTime = cTime

        cv.putText(img, f"FPS : {int(fps)}", (25, 140), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)

        # Number of Hands
        cv.putText(img, f"No. Of Hands : {len(multi_hand_landmarks)}", (25, 180), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        cv.imshow("Gesture Detector", img)

        key = cv.waitKey(1)

        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()