import math
import numpy as np
from collections import deque, Counter

class GestureDetector:
    def __init__(self, history_length=10, confidence_threshold=0.7):
        """
        Initializes the state machine and temporal buffers.
        """
        self.history_length = history_length
        self.confidence_threshold = confidence_threshold
        self.history = deque(maxlen=history_length)
        
        # Landmarks reference
        self.WRIST = 0
        self.THUMB = [1, 2, 3, 4]
        self.TIPS = [8, 12, 16, 20]
        self.DIPS = [7, 11, 15, 19]
        self.PIPS = [6, 10, 14, 18]
        self.MCPS = [5, 9, 13, 17]

    # HELPER FUNCTIONS (Feature Extraction)
    
    def _get_xy(self, lmList, point):

        if isinstance(point, (int, np.integer)):
            point = lmList[int(point)]
        return point[1], point[2]

    def distance(self, lmList, p1, p2):

        x1, y1 = self._get_xy(lmList, p1)
        x2, y2 = self._get_xy(lmList, p2)
        return math.hypot(x2 - x1, y2 - y1)
    
    def getHandSize(self, lmList):

        return max(
            self.distance(lmList, 0, 9),
            self.distance(lmList, 5, 17)
        )

    def normalized_distance(self, p1, p2, lmList):

        return (
            self.distance(lmList, p1, p2)
            /
            (self.getHandSize(lmList) + 1e-6)
        )
    
    def getAngle(self, p1, p2, p3):
        """
        Returns angle (degrees) at p2 formed by:
        p1 -> p2 -> p3
        """
        a = np.array([p1[1], p1[2]])
        b = np.array([p2[1], p2[2]])
        c = np.array([p3[1], p3[2]])

        ba = a - b
        bc = c - b

        cosine = np.dot(ba, bc) / (
            np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
        )

        cosine = np.clip(cosine, -1.0, 1.0)

        return np.degrees(np.arccos(cosine))
    
    def getFingerAngle(self, lmList, mcp, pip, dip, tip):
        angle1 = self.getAngle(
            lmList[mcp],
            lmList[pip],
            lmList[dip]
        )

        angle2 = self.getAngle(
            lmList[pip],
            lmList[dip],
            lmList[tip]
        )

        return (angle1 + angle2) / 2
    
    def getThumbState(self, lmList):

        angle1 = self.getAngle(
            lmList[1],
            lmList[2],
            lmList[3]
        )

        angle2 = self.getAngle(
            lmList[2],
            lmList[3],
            lmList[4]
        )

        thumb_angle = (angle1 + angle2) / 2

        tip_dist = self.normalized_distance(
            self.WRIST,
            self.THUMB[3],
            lmList
        )

        mcp_dist = self.normalized_distance(
            self.WRIST,
            2,
            lmList
        )

        ratio = tip_dist / (mcp_dist + 1e-6)

        if thumb_angle > 145 and ratio > 1.20:       
            return 2

        elif thumb_angle < 125 and ratio < 1.10:
            return 0

        return 1

    def getFingerStates(self, lmList):
        """
        Returns:
            [thumb, index, middle, ring, pinky]

            0 -> Curled
            1 -> Half Curled
            2 -> Extended
        """
        # mcp  = Metacarpophalangeal joint (knuckle)
        # pip  = Proximal Interphalangeal joint (middle joint)
        # dip  = Distal Interphalangeal joint (joint near fingertip)
        # tip  = Fingertip

        finger_landmarks = [
        #   mcp, pip, dip, tip
            (5, 6, 7, 8),      # Index
            (9, 10, 11, 12),   # Middle
            (13, 14, 15, 16),  # Ring
            (17, 18, 19, 20)   # Pinky
        ]

        states = []
        states.append(self.getThumbState(lmList))     

        for mcp, pip, dip, tip in finger_landmarks:

            finger_angle = self.getFingerAngle(
                lmList,
                mcp,
                pip,
                dip,
                tip
            )

            tip_dist = self.normalized_distance(
                lmList[self.WRIST],
                lmList[tip],
                lmList
            )

            pip_dist = self.normalized_distance(
                lmList[self.WRIST],
                lmList[pip],
                lmList
            )

            ratio = tip_dist / (pip_dist + 1e-6)
            
            if ( finger_angle > 155 and
                    ratio > 1.15
                ):
                state = 2

            elif ( finger_angle < 125 and
                ratio < 1.10
            ):
                state = 0

            else:
                state = 1

            states.append(state)

        return states
    
    def isPinching(self, lmList, finger1=4, finger2=8, threshold=0.35):
        """
        Checks if two fingertips are touching.
        Default: Thumb tip (4) and Index tip (8).
        """
        dist = self.normalized_distance(
            finger1,
            finger2,
            lmList
        )

        return dist < threshold

    def isFingerExtended(self, state):
        return state == 2

    def isFingerCurled(self, state):
        return state == 0

    def isFingerHalfCurled(self, state):
        return state == 1
    
    # SINGLE HAND GESTURES
    
    def isOpenPalm(self, states):
        return all(
            self.isFingerExtended(state)
            for state in states
        )
    
    def isFist(self, states):
        return (
            (self.isFingerCurled(states[0]) or
            self.isFingerHalfCurled(states[0]))
            and
            all(
                self.isFingerCurled(state)
                for state in states[1:]
            )
        )
    
    def isThumbsUp(self, states, lmList):

        if not (
            self.isFingerExtended(states[0]) and
            all(
                self.isFingerCurled(state)
                for state in states[1:]
            )
        ):
            return False

        return (
            lmList[4][2] <
            lmList[3][2] <
            lmList[2][2]
        )
    
    def isThumbsDown(self, states, lmList):

        if not (
            self.isFingerExtended(states[0]) and
            all(
                self.isFingerCurled(state)
                for state in states[1:]
            )
        ):
            return False

        return (
            lmList[4][2] >
            lmList[3][2] >
            lmList[2][2]
        )
    
    def isOK(self, states, lmList):

        if not (
            self.isFingerExtended(states[2]) and
            self.isFingerExtended(states[3]) and
            self.isFingerExtended(states[4])
        ):
            return False

        return self.isPinching(lmList)
    
    def isRock(self, states, lmList):

        if not (
            self.isFingerExtended(states[1]) and
            self.isFingerExtended(states[4])
        ):
            return False

        if self.isFingerExtended(states[2]):
            return False

        if self.isFingerExtended(states[3]):
            return False

        return (
            self.normalized_distance(4, 17, lmList)
            <
            self.normalized_distance(4, 5, lmList) + 0.1
        )
    
    def isSpiderman(self, states, lmList):

        if not self.isFingerExtended(states[0]):
            return False

        if not (
            self.isFingerExtended(states[1]) and
            self.isFingerExtended(states[4])
        ):
            return False

        if self.isFingerExtended(states[2]):
            return False

        if self.isFingerExtended(states[3]):
            return False

        return (
            self.normalized_distance(4, 17, lmList)
            >
            self.normalized_distance(4, 5, lmList)
        )

    def isMiddleFinger(self, states, lmList):

        if not self.isFingerExtended(states[2]):
            return False

        if self.isFingerExtended(states[0]):
            return False

        if self.isFingerExtended(states[1]):
            return False

        if self.isFingerExtended(states[3]):
            return False

        if self.isFingerExtended(states[4]):
            return False

        return (
            self.normalized_distance(4, 17, lmList)
            <
            self.normalized_distance(4, 5, lmList) + 0.1
        )
    
    def isNumberOne(self, states):
        return (
            not self.isFingerExtended(states[0]) and
            self.isFingerExtended(states[1]) and
            not self.isFingerExtended(states[2]) and
            not self.isFingerExtended(states[3]) and
            not self.isFingerExtended(states[4])
        )
    
    def isNumberTwo(self, states):
        return (
            self.isFingerExtended(states[1]) and
            self.isFingerExtended(states[2]) and
            not self.isFingerExtended(states[3]) and
            not self.isFingerExtended(states[4])
        )
    
    def isNumberThree(self, states):
        return (
            self.isFingerExtended(states[1]) and
            self.isFingerExtended(states[2]) and
            self.isFingerExtended(states[3]) and
            not self.isFingerExtended(states[4])
        )
    
    def isNumberFour(self, states):
        return (
            not self.isFingerExtended(states[0]) and
            self.isFingerExtended(states[1]) and
            self.isFingerExtended(states[2]) and
            self.isFingerExtended(states[3]) and
            self.isFingerExtended(states[4])
        )
    
    def isFingerHeart(self, states, lmList):

        if not self.isFingerExtended(states[0]):
            return False

        if self.isFingerExtended(states[1]):
            return False

        if any(
            self.isFingerExtended(states[i])
            for i in [2, 3, 4]
        ):
            return False

        return (
            self.normalized_distance(3, 8, lmList) < 0.35
        )
    
    def isNumberFive(self, states):
        return self.isOpenPalm(states)
    
    def detectSingleHand(self, lmList):
        """Evaluates single hand landmarks and returns string gesture."""

        states = self.getFingerStates(lmList)

        # Special gestures first
        if self.isSpiderman(states, lmList):
            return "Spider-Man"

        if self.isRock(states, lmList):
            return "Rock"

        if self.isMiddleFinger(states, lmList):
            return "Fuck Off"

        if self.isOK(states, lmList):
            return "OK"

        if self.isThumbsUp(states, lmList):
            return "Thumbs Up"

        if self.isThumbsDown(states, lmList):
            return "Thumbs Down"
        
        if self.isFingerHeart(states,lmList):
            return "Finger Heart"

        # Numbers
        if self.isNumberFive(states):
            return "Five / Open Palm"

        if self.isNumberFour(states):
            return "Four"

        if self.isNumberThree(states):
            return "Three"

        if self.isNumberTwo(states):
            return "Two / Victory"

        if self.isNumberOne(states):
            return "One"

        # Basic gesture
        if self.isFist(states):
            return "Fist"
        
        return "Unknown Gesture"
    
    # TWO HAND GESTURES
    
    def isNumberSix(self, g1, g2):

        hands = {g1, g2}

        if hands == {"Five / Open Palm", "One"}:
            return True

        if g1 == "Three" and g2 == "Three":
            return True

        if hands == {"Four", "Two / Victory"}:
            return True
        
        return False
        

    def isNumberSeven(self, g1, g2):

        hands = {g1, g2}

        if hands == {"Five / Open Palm", "Two / Victory"}:
            return True

        if hands == {"Four", "Three"}:
            return True

        return False

    def isNumberEight(self, g1, g2):

        hands = {g1, g2}

        if hands == {"Five / Open Palm", "Three"}:
            return True

        if g1 == "Four" and g2 == "Four":
            return True

        return False

    def isNumberNine(self, g1, g2):

        hands = {g1, g2}

        if hands == {"Five / Open Palm", "Four"}:
            return True

        return False

    def isNumberTen(self, g1, g2):

        return (
            g1 == "Five / Open Palm" and
            g2 == "Five / Open Palm"
        )

    def isNumberEleven(self, g1, g2):

        return (
            g1 == "One" and
            g2 == "One"
        )
    
    def interHandDistance(self, lmList1, p1, lmList2, p2):

        x1, y1 = lmList1[p1][1], lmList1[p1][2]
        x2, y2 = lmList2[p2][1], lmList2[p2][2]
        return math.hypot(x2 - x1, y2 - y1)

    def isNamaste(self, lmList1, lmList2):

        s1 = self.getFingerStates(lmList1)
        s2 = self.getFingerStates(lmList2)

        if not (
            self.isOpenPalm(s1) and
            self.isOpenPalm(s2)
        ):
            return False

        hand_size = max(
            self.getHandSize(lmList1),
            self.getHandSize(lmList2)
        )

        threshold = 0.30

        for i in range(21):
            dist = (
                self.interHandDistance(lmList1, i, lmList2, i)
                / (hand_size + 1e-6)
            )

            if dist > threshold:
                return False

        return True

    def isPossibleNamasteSingleHand(self, lmList):

        states = self.getFingerStates(lmList)

        if not self.isOpenPalm(states):
            return False

        xs = [pt[1] for pt in lmList]
        ys = [pt[2] for pt in lmList]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        hand_size = self.getHandSize(lmList)

        width_ratio = width / (hand_size + 1e-6)
        height_ratio = height / (hand_size + 1e-6)

        return (
            width_ratio < 0.8 and
            height_ratio > 2.0
        )

    def isLove(self, lmList1, lmList2):

        s1 = self.getFingerStates(lmList1)
        s2 = self.getFingerStates(lmList2)

        for i in [1, 2, 3, 4]:
            if (
                self.isFingerExtended(s1[i]) or
                self.isFingerExtended(s2[i])
            ):
                return False

        hand_size = max(
            self.getHandSize(lmList1),
            self.getHandSize(lmList2)
        )

        thumb_dist = (
            self.interHandDistance(lmList1, 4, lmList2, 4)
            / (hand_size + 1e-6)
        )

        index_dist = (
            self.interHandDistance(lmList1, 8, lmList2, 8)
            / (hand_size + 1e-6)
        )

        middle_dist = (
            self.interHandDistance(lmList1, 12, lmList2, 12)
            / (hand_size + 1e-6)
        )

        return (
            thumb_dist < 0.35 and
            index_dist < 0.35 and
            middle_dist < 0.35
        )

    def isFuckOff(self, lmList1, lmList2):
        return (
        self.detectSingleHand(lmList1) == "Fuck Off" and
        self.detectSingleHand(lmList2) == "Fuck Off"
        )

    def isDoubleRock(self, lmList1, lmList2):
        return (
        self.detectSingleHand(lmList1) == "Rock" and
        self.detectSingleHand(lmList2) == "Rock"
        )
    
    def isDoubleFingerHeart(self, lmList1, lmList2):
        return (
        self.detectSingleHand(lmList1) == "Finger Heart" and
        self.detectSingleHand(lmList2) == "Finger Heart"
        )
    
    def isDoubleSpiderMan(self, lmList1, lmList2):
        return (
        self.detectSingleHand(lmList1) == "Spider-Man" and
        self.detectSingleHand(lmList2) == "Spider-Man"
        )

    def isDoubleVictory(self, lmList1, lmList2):
        return (
        self.detectSingleHand(lmList1) == "Two / Victory" and
        self.detectSingleHand(lmList2) == "Two / Victory"
        )

    def isDoubleThumbsUp(self, lmList1, lmList2):
        return (
        self.detectSingleHand(lmList1) == "Thumbs Up" and
        self.detectSingleHand(lmList2) == "Thumbs Up"
        )

    def isDoubleThumbsDown(self, lmList1, lmList2):
        return (
        self.detectSingleHand(lmList1) == "Thumbs Down" and
        self.detectSingleHand(lmList2) == "Thumbs Down"
        )

    def isDoubleFist(self, lmList1, lmList2):
        return (
        self.detectSingleHand(lmList1) == "Fist" and
        self.detectSingleHand(lmList2) == "Fist"
        )

    def isSushantOK(self, lmList1, lmList2):
        return (
        self.detectSingleHand(lmList1) == "OK" and
        self.detectSingleHand(lmList2) == "OK"
        )

    def detectTwoHands(self, lmList1, lmList2):

        # Special two-hand gestures
        if self.isNamaste(lmList1, lmList2):
            return "Namaste"

        if self.isLove(lmList1, lmList2):
            return "Love You :)"
        
        if self.isFuckOff(lmList1, lmList2):
            return "Fuck Off Motherfucker"

        if self.isDoubleRock(lmList1, lmList2):
            return "Cool :)"
        
        if self.isDoubleSpiderMan(lmList1,lmList2):
            return "Spider-Man"

        if self.isDoubleVictory(lmList1, lmList2):
            return "Peace :)"

        if self.isDoubleThumbsUp(lmList1, lmList2):
            return "Good :)"

        if self.isDoubleThumbsDown(lmList1, lmList2):
            return "Bad :("

        if self.isSushantOK(lmList1, lmList2):
            return "Sushant Singh's OK"
        
        if self.isDoubleFist(lmList1,lmList2):
            return "Punch"
        
        if self.isDoubleFingerHeart(lmList1,lmList2):
            return "Finger Heart"

        # Numbers
        g1 = self.detectSingleHand(lmList1)
        g2 = self.detectSingleHand(lmList2)

        if self.isNumberSix(g1, g2):
            return "Number Six"

        if self.isNumberSeven(g1, g2):
            return "Number Seven"

        if self.isNumberEight(g1, g2):
            return "Number Eight"

        if self.isNumberNine(g1, g2):
            return "Number Nine"

        if self.isNumberTen(g1, g2):
            return "Number Ten"

        if self.isNumberEleven(g1, g2):
            return "Number Eleven"

        return f"{g1} & {g2}"
    
    # TEMPORAL STABILITY / STATE MACHINE

    def process_frame(self, multi_hand_landmarks):

        if not multi_hand_landmarks:
            self.history.append("UNKNOWN")
            return self.get_stable_gesture()
        
        if len(multi_hand_landmarks) == 1:

            lm = multi_hand_landmarks[0]

            if self.isPossibleNamasteSingleHand(lm):
                raw_gesture = "Namaste"
            else:
                raw_gesture = self.detectSingleHand(lm)

        elif len(multi_hand_landmarks) == 2:
            raw_gesture = self.detectTwoHands(
                multi_hand_landmarks[0],
                multi_hand_landmarks[1]
            )

        else:
            raw_gesture = "UNKNOWN"

        self.history.append(raw_gesture)

        return self.get_stable_gesture()
    
    def get_stable_gesture(self):
        
        """
        Returns the most stable gesture from the history buffer.
        """

        if len(self.history) < self.history_length:
            return "UNKNOWN"

        counter = Counter(self.history)
        gesture, count = counter.most_common(1)[0]

        confidence = count / len(self.history)

        if ( gesture != "UNKNOWN"
            and confidence >= self.confidence_threshold ):
            return gesture

        return "UNKNOWN"