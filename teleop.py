import argparse
import math
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pygame

from boxpush import push

# add command line arguments for if you want to use mediapipe or keyboard control
def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperate the box pushing task.")
    parser.add_argument(
        "--control",
        choices=["keyboard", "mediapipe"],
        default="keyboard",
        help="Select control modality.",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of rollouts to collect.")
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=["human", "none"],
        help="Rendering mode passed to the environment.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for MediaPipe control.")
    parser.add_argument(
        "--pinch-min",
        type=float,
        default=0.15,
        help="Distance (normalized) treated as zero force for pinch control.",
    )
    parser.add_argument(
        "--pinch-max",
        type=float,
        default=0.5,
        help="Distance (normalized) treated as max force for pinch control.",
    )
    parser.add_argument(
        "--show-hand-debug",
        action="store_true",
        help="Display annotated camera feed when using MediaPipe.",
    )
    return parser.parse_args()


def update_action_from_keys(buf: np.ndarray):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        buf[0] = -1.0
    elif keys[pygame.K_RIGHT]:
        buf[0] = 1.0
    else:
        buf[0] = 0.0


class MediaPipeHandController:
    """Maps thumb-index distance to env action using MediaPipe Hands."""

    def __init__(self, camera_index: int, min_dist: float, max_dist: float, max_force: float, show_debug: bool):
        self._min_dist = min_dist
        self._max_dist = max(min_dist + 1e-3, max_dist)
        self._max_force = max_force
        self._show_debug = show_debug
        self._window_name = "MediaPipe Teleop"

        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}")

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._cached_action = np.zeros(1, dtype=np.float32)
        self._active = True
        self._last_force = 0.0

    @staticmethod
    def _euclidean_dist(lm1, lm2):
        dx = lm1.x - lm2.x
        dy = lm1.y - lm2.y
        dz = lm1.z - lm2.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def get_action(self) -> np.ndarray:
        if not self._active:
            return self._cached_action

        success, frame = self._cap.read()
        if not success:
            self._cached_action[0] = 0.0
            return self._cached_action

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            self._cached_action[0] = 0.0
            self._last_force = 0.0
            display_frame = frame
        else:
            hand_landmarks = results.multi_hand_landmarks[0]
            thumb_tip = hand_landmarks.landmark[self._mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self._mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dist = self._euclidean_dist(thumb_tip, index_tip)
            #this is what maps the distance to the force
            norm = (dist - self._min_dist) / (self._max_dist - self._min_dist)
            norm = float(np.clip(norm, 0.0, 1.0))

            handed_label = "Right"
            if results.multi_handedness:
                handed_label = results.multi_handedness[0].classification[0].label

            direction = 1.0 if handed_label == "Right" else -1.0
            signed_force = direction * norm * self._max_force
            self._cached_action[0] = signed_force / self._max_force
            self._last_force = signed_force
            display_frame = frame

        if self._show_debug:
            annotated = display_frame.copy()
            if results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated, results.multi_hand_landmarks[0], self._mp_hands.HAND_CONNECTIONS
                )
            cv2.putText(
                annotated,
                f"Force: {self._last_force:+.2f} N",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Force: {self._last_force:+.2f} N",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(self._window_name, annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                self._active = False

        return self._cached_action

    def is_active(self) -> bool:
        return self._active

    def close(self):
        self._hands.close()
        self._cap.release()
        if self._show_debug:
            cv2.destroyWindow(self._window_name)


def run_keyboard(args) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    pygame.init()
    
    env = push(render_mode=args.render_mode)
    demos_obs: List[np.ndarray] = []
    demos_act: List[np.ndarray] = []
    cached_action = np.zeros(1, dtype=np.float32)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            print(f"Episode {ep+1}/{args.episodes} - target={env.x_target:.3f}")

            pygame.event.pump()
            update_action_from_keys(cached_action)

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return demos_obs, demos_act

                update_action_from_keys(cached_action)
                next_obs, reward, terminated, truncated, info = env.step(cached_action)
                done = terminated or truncated

                demos_obs.append(obs)
                demos_act.append(cached_action.copy())
                obs = next_obs

            print("Episode end")
    finally:
        env.close()
        pygame.quit()

    return demos_obs, demos_act


def run_mediapipe(args) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    env = push(render_mode=args.render_mode)
    controller = MediaPipeHandController(
        camera_index=args.camera_index,
        min_dist=args.pinch_min,
        max_dist=args.pinch_max,
        max_force=env.Fmax,
        show_debug=args.show_hand_debug,
    )

    demos_obs: List[np.ndarray] = []
    demos_act: List[np.ndarray] = []

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            print(f"Episode {ep+1}/{args.episodes} - target={env.x_target:.3f}")

            while not done and controller.is_active():
                action = controller.get_action()
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                demos_obs.append(obs)
                demos_act.append(action.copy())
                obs = next_obs

            if not controller.is_active():
                print("MediaPipe control stopped by user.")
                break

            print("Episode end")
    finally:
        controller.close()
        env.close()

    return demos_obs, demos_act


def save_demos(demos_obs: List[np.ndarray], demos_act: List[np.ndarray]):
    if not demos_obs or not demos_act:
        return
    demos_obs_arr = np.stack(demos_obs)
    demos_act_arr = np.stack(demos_act)
    np.save("demos_obs.npy", demos_obs_arr)
    np.save("demos_act.npy", demos_act_arr)
    print("Saved demos to demos_obs.npy and demos_act.npy")


def main():
    args = parse_args()

    if args.control == "mediapipe":
        demos_obs, demos_act = run_mediapipe(args)
    else:
        demos_obs, demos_act = run_keyboard(args)

    save_demos(demos_obs, demos_act)


if __name__ == "__main__":
    main()
