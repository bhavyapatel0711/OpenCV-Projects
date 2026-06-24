# Gesture Controlled Volume

Control your system volume using hand gestures through Computer Vision and Hand Tracking. This project detects the distance between your thumb and index finger and maps it to the system's volume level in real time.

> **Note:** This project uses the MediaPipe Solution API (`mp.solutions.hands`) and was developed using **Python 3.11.x**. It may not work correctly with newer MediaPipe releases or Python versions.

---

## Features

* Real-time hand tracking using MediaPipe.
* Gesture-based volume control.
* Camera distance independent using normalized hand measurements.
* Smooth volume transitions and animated volume bar.
* Live FPS display.

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* Pycaw

---

## Requirements

* Python 3.11.x
* Webcam
* MediaPipe Solution API (`mp.solutions`) is used in this project.

> **Note:** Newer versions of MediaPipe or Python may not be compatible with this implementation.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python Gesture_Controlled_Volume.py
```

Press **Q** to quit the application.

---

## How It Works

1. Capture frames from the webcam.
2. Detect the hand and extract landmarks.
3. Get the coordinates of the thumb tip and index finger tip.
4. Calculate the normalized distance between the fingertips.
5. Map the distance to the system volume range.
6. Smooth the output to avoid sudden jumps.
7. Update the system volume and display the UI.

---

## Demo Video

[Watch Demo](demo.mp4)

---

## Project Structure

```text
Gesture-Controlled-Volume/
│
├── Gesture_Controlled_Volume.py
├── Hand_Tracking_Module.py
├── requirements.txt
├── demo.mp4
└── README.md
```
