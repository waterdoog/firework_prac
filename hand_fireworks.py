import math
import random
import time
from collections import defaultdict
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2


class FireworkParticle:
    """Single particle that makes up a firework burst."""

    def __init__(self, origin: Tuple[int, int], velocity: Tuple[float, float], color: Tuple[int, int, int], lifetime: float) -> None:
        self.position = np.array(origin, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.color = color
        self.lifetime = lifetime
        self.remaining = lifetime

    def update(self, dt: float) -> None:
        gravity = 520.0  # pixels per second squared
        self.velocity[1] += gravity * dt
        self.position += self.velocity * dt
        self.remaining -= dt

    def is_alive(self) -> bool:
        return self.remaining > 0

    def draw(self, frame: np.ndarray) -> None:
        if not self.is_alive():
            return
        height, width = frame.shape[:2]
        x = int(self.position[0])
        y = int(self.position[1])
        if x < 0 or x >= width or y < 0 or y >= height:
            return
        fade = max(0.05, self.remaining / self.lifetime)
        radius = max(1, int(3 * fade + 1))
        color = tuple(int(channel * fade + 40 * (1 - fade)) for channel in self.color)
        cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


class Firework:
    """Group of particles that originate from a single open-hand event."""

    def __init__(self, origin: Tuple[int, int]) -> None:
        self.particles = self._create_particles(origin)

    def _create_particles(self, origin: Tuple[int, int]):
        palette = [
            (255, 220, 95),
            (255, 125, 90),
            (130, 200, 255),
            (180, 255, 180),
            (255, 140, 220),
        ]
        particles = []
        count = random.randint(60, 90)
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(140, 260)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - random.uniform(80, 160)
            lifetime = random.uniform(0.9, 1.6)
            color = random.choice(palette)
            particles.append(FireworkParticle(origin, (vx, vy), color, lifetime))
        return particles

    def update(self, dt: float) -> None:
        for particle in self.particles:
            particle.update(dt)
        self.particles = [particle for particle in self.particles if particle.is_alive()]

    def draw(self, frame: np.ndarray) -> None:
        for particle in self.particles:
            particle.draw(frame)

    def is_alive(self) -> bool:
        return len(self.particles) > 0


mp_hands = mp.solutions.hands
DRAWING_UTILS = mp.solutions.drawing_utils


def is_hand_open(hand_landmarks: landmark_pb2.NormalizedLandmarkList, handedness: str) -> bool:
    landmarks = hand_landmarks.landmark

    finger_tip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]

    finger_pip_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]

    extended = 0
    for tip_id, pip_id in zip(finger_tip_ids, finger_pip_ids):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            extended += 1

    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    if handedness == "Right":
        if thumb_tip.x > thumb_ip.x:
            extended += 1
    else:
        if thumb_tip.x < thumb_ip.x:
            extended += 1

    return extended >= 4


def get_hand_anchor(hand_landmarks: landmark_pb2.NormalizedLandmarkList, frame_shape: Tuple[int, int, int]) -> Tuple[int, int]:
    height, width = frame_shape[:2]
    anchor = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    x = int(anchor.x * width)
    y = int(anchor.y * height)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Make sure a webcam is connected.")

    fireworks: list[Firework] = []
    prev_time = time.perf_counter()
    max_fireworks = 12
    hand_cooldowns: defaultdict[str, float] = defaultdict(float)

    cv2.namedWindow("Gesture Fireworks", cv2.WINDOW_NORMAL)

    with mp_hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed: check that your webcam is not in use by another application.")
                time.sleep(0.5)
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            results = hands.process(frame_rgb)
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            dt = min(dt, 0.05)
            for key in list(hand_cooldowns.keys()):
                hand_cooldowns[key] = max(0.0, hand_cooldowns[key] - dt)

            triggered_hands: list[Tuple[str, Tuple[int, int]]] = []

            frame.flags.writeable = True
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                    DRAWING_UTILS.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    label = hand_info.classification[0].label
                    index = hand_info.classification[0].index
                    hand_key = f"{label}_{index}"
                    if is_hand_open(hand_landmarks, label):
                        position = get_hand_anchor(hand_landmarks, frame.shape)
                        triggered_hands.append((hand_key, position))

            for hand_key, hand_position in triggered_hands:
                if hand_cooldowns[hand_key] <= 0.0:
                    fireworks.append(Firework(hand_position))
                    if len(fireworks) > max_fireworks:
                        fireworks = fireworks[-max_fireworks:]
                    hand_cooldowns[hand_key] = 0.6

            for firework in fireworks:
                firework.update(dt)
                firework.draw(frame)
            fireworks = [firework for firework in fireworks if firework.is_alive()]

            cv2.putText(
                frame,
                "Open your hand to launch fireworks  |  Press ESC to exit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Gesture Fireworks", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
