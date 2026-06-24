# Bicep Curl Tracker

A real-time bicep curl counter built using Computer Vision and Pose Estimation. This project tracks body landmarks, calculates elbow joint angles, and counts repetitions for both arms independently.

> **Note:** This project uses the MediaPipe Solution API (`mp.solutions.pose`) and was developed using **Python 3.11.x**. It may not work correctly with newer MediaPipe releases or Python versions.

---

## Features

* Real-time human pose tracking using MediaPipe.
* Independent left and right arm repetition counting.
* Elbow angle calculation for curl detection.
* Automatic rep counting using movement direction.
* Live FPS display.
* Professional dashboard displaying repetition counts.

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy

---

## Requirements

* Python 3.11.x
* Webcam
* MediaPipe Solution API (`mp.solutions`) is used in this project.

> **Note:** Newer versions of MediaPipe or Python may not be compatible with this implementation.

Install the dependencies using:

```bash id="0w7wxn"
pip install -r requirements.txt
```

---

## Run

```bash id="c68hsy"
python Bicep_Curl_Tracker.py
```

Press **Q** to quit the application.

---

## How It Works

1. Capture frames from the webcam.
2. Detect body landmarks using MediaPipe Pose.
3. Extract shoulder, elbow, and wrist coordinates.
4. Calculate the elbow angle using geometric relationships.
5. Map the angle to the arm's contraction percentage.
6. Track movement direction (up and down).
7. Count repetitions when a complete curl cycle is detected.
8. Display repetition counts and FPS in real time.

---

## Demo Video

[Watch Demo](demo.mp4)

---

## Project Structure

```text id="gtm5d6"
Bicep-Curl-Tracker/
│
├── Bicep_Curl_Tracker.py
├── Pose_Tracking_Module.py
├── requirements.txt
├── demo.mp4
└── README.md
```
