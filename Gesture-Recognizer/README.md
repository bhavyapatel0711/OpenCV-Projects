# Real-Time Hand Gesture Recognition System

A modular real-time hand gesture recognition system built using **OpenCV** and **MediaPipe** that recognizes both **single-hand** and **two-hand** gestures using landmark geometry, finger state estimation, distance normalization, and temporal smoothing techniques.

> **Note:** This project uses the MediaPipe Solution API (`mp.solutions.hands`) and was developed using **Python 3.11.x**. It may not work correctly with newer MediaPipe releases or Python versions.

---

## Features

* Real-time hand tracking and landmark detection.
* Recognition of both single-hand and two-hand gestures.
* Finger state estimation (Curled, Half-Curled, Extended).
* Distance normalization for camera-distance invariance.
* Angle-based gesture classification.
* Temporal smoothing using a confidence-based history buffer.
* Modular architecture allowing easy addition of new gestures.
* Real-time HUD displaying the detected gesture, FPS, and number of hands.

---

## Supported Gestures

### Single Hand Gestures

| Gesture          | Description                     |
| ---------------- | ------------------------------- |
| Open Palm / Five | All fingers extended            |
| Fist             | All fingers curled              |
| Thumbs Up        | Thumb extended upward           |
| Thumbs Down      | Thumb extended downward         |
| OK               | Thumb and index finger touching |
| Rock             | Index and pinky extended        |
| Spider-Man       | Thumb, index and pinky extended |
| One              | Index finger only               |
| Two / Victory    | Index and middle fingers        |
| Three            | Index, middle and ring fingers  |
| Four             | Four fingers extended           |
| Finger Heart     | Korean finger heart gesture     |
| Middle Finger    | Middle finger only              |

---

### Two Hand Gestures

| Gesture              | Description                            |
| -------------------- | -------------------------------------- |
| Number Six           | Combination of gestures representing 6 |
| Number Seven         | Combination of gestures representing 7 |
| Number Eight         | Combination of gestures representing 8 |
| Number Nine          | Combination of gestures representing 9 |
| Number Ten           | Two open palms                         |
| Number Eleven        | One gesture on both hands              |
| Namaste              | Joined palms gesture                   |
| Love You :)          | Heart shape formed using both hands    |
| Cool :)              | Double rock gesture                    |
| Peace :)             | Double victory gesture                 |
| Good :)              | Double thumbs up                       |
| Bad :(               | Double thumbs down                     |
| Punch                | Double fists                           |
| Sushant Singh's OK   | Double OK gesture                      |
| Spider-Man           | Double Spider-Man gesture              |
| Finger Heart         | Double finger heart gesture            |
| Double Middle Finger | Dual middle finger gesture             |

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

```bash
pip install -r requirements.txt
```

## Run

```bash
python Gesture_Recognizer.py
```

Press **Q** to quit the application.

---

## Demo Video

[▶️ Watch Demo](demo.mp4)

---

## Project Architecture

```text
Webcam Frame
      ↓
Hand Detection (MediaPipe)
      ↓
Landmark Extraction
      ↓
Feature Extraction
      ↓
Finger State Estimation
      ↓
Single-Hand / Two-Hand Classification
      ↓
Temporal Smoothing
      ↓
Final Gesture Prediction
      ↓
Display Output
```

---

## How It Works

### 1. Hand Detection

MediaPipe Hands detects **21 landmarks** for each hand in real time.

---

### 2. Finger State Estimation

Each finger is classified into one of three states:

```text
0 → Curled
1 → Half Curled
2 → Extended
```

Finger states are determined using:

* Joint angles
* Distance ratios
* Relative landmark positions

---

### 3. Distance Normalization

The apparent size of the hand changes depending on its distance from the camera. To make the detector robust, all important distances are normalized.

```text
Normalized Distance =
Euclidean Distance / Hand Size
```

Hand size is estimated using:

```text
max(
    Wrist → Middle MCP,
    Index MCP → Pinky MCP
)
```

This makes the detector nearly independent of the user's distance from the camera.

---

### 4. Angle-Based Feature Extraction

Finger joint angles are computed using geometric relationships between landmarks.

These angles help distinguish between:

* Curled fingers
* Half-curled fingers
* Fully extended fingers

---

### 5. Gesture Classification

Each gesture is defined using a combination of:

* Finger states
* Joint angles
* Relative fingertip distances
* Inter-hand relationships

This rule-based approach provides interpretable and reliable gesture recognition.

---

### 6. Temporal Stability

Frame-by-frame predictions can fluctuate due to landmark noise. To reduce flickering, a history buffer stores previous predictions.

```python
deque(maxlen=history_length)
```

The final gesture is selected using majority voting.

```text
confidence = count / history_length
```

A gesture is accepted only if:

```text
confidence ≥ confidence_threshold
```

This significantly improves prediction stability.

---

## Core Algorithms Used

* Euclidean Distance
* Distance Normalization
* Angle Computation using Vector Geometry
* Finger State Estimation
* Rule-Based Gesture Classification
* Temporal Smoothing
* Majority Voting

---

## Challenges Faced

* Distinguishing between visually similar gestures such as Rock and Spider-Man.
* Designing robust rules for two-hand gestures.
* Making the detector invariant to hand size and camera distance.
* Reducing prediction flickering caused by frame-to-frame landmark noise.
* Handling partially folded fingers and ambiguous poses.
* Building a modular system that can easily be extended with new gestures.

---

## Milestones Achieved

* Developed a custom hand tracking module.
* Implemented finger state estimation using geometric features.
* Added support for both single-hand and two-hand gestures.
* Implemented distance normalization for scale invariance.
* Added temporal smoothing using a confidence-based history buffer.
* Successfully recognized 25+ gestures in real time.
* Built a modular gesture recognition framework that can be extended with additional gestures.

---

## Future Scope

* Dynamic gesture recognition (wave, swipe, zoom).
* Gesture-controlled applications and games.
* Sign language recognition.
* Dataset generation for machine learning models.
* Deep learning based gesture classification.
* Multi-user gesture interaction.

---

## Project Structure

```text
Gesture-Recognition-System/
│
├── Gesture_Recognizer.py
├── Gesture_Detection_Module.py
├── Hand_Tracking_Module.py
├── requirements.txt
├── demo.mp4
└── README.md
```

