import argparse
import math
from typing import List, Tuple
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame
import torch
import torch.nn as nn

from boxpush import push


class BehaviorCloningPolicy(nn.Module):
    """Simple MLP with tanh output to mimic teleop actions."""

    def __init__(self, obs_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs)
        return torch.tanh(out)


# add command line arguments for if you want to use mediapipe or keyboard control
# this also allows us to change the amount of episodes to run, and other parameters
# for mediapipe teleop, we wanna run <python teleop.py --control mediapipe --show-hand-debug> 
# for visualizing the trained policy, we wanna run <python teleop.py --control policy --policy-path policy.pt> 
def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperate the box pushing task.")
    parser.add_argument(
        "--control",
        choices=["keyboard", "mediapipe", "policy"],
        default="keyboard",
        help="Select control modality.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of rollouts to collect.")
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
        default=0.1,
        help="Distance (normalized) treated as zero force for pinch control.",
    )
    parser.add_argument(
        "--pinch-max",
        type=float,
        default=0.4,
        help="Distance (normalized) treated as max force for pinch control.",
    )
    parser.add_argument(
        "--show-hand-debug",
        action="store_true",
        help="Display annotated camera feed when using MediaPipe.",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="policy.pt",
        help="Path to a trained behavior cloning policy checkpoint.",
    )
    return parser.parse_args()

# Here is where we do the keyboard control using pygame
# The force values are either -1, 0, or 1 and scaled to the mac force.
# There is some delay to response time, but it will do for now
def update_action_from_keys(buf: np.ndarray):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        buf[0] = -1.0
    elif keys[pygame.K_RIGHT]:
        buf[0] = 1.0
    else:
        buf[0] = 0.0

# This is the important class that allows us to use the distance between thumb and index finger to map to force output
# Here we have a more variable force output, instead of just -1, 0, or 1
class MediaPipeHandController:
    """Maps thumb-index distance to env action using MediaPipe Hands."""

    def __init__(self, camera_index: int, min_dist: float, max_dist: float, max_force: float, show_debug: bool):
        self._min_dist = min_dist
        self._max_dist = max_dist
        self._max_force = max_force
        self._show_debug = show_debug
        self._window_name = "MediaPipe Teleop"

        self._cap = cv2.VideoCapture(camera_index) #get camera input
        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}")

        self._mp_hands = mp.solutions.hands 
        self._mp_drawing = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands( # only takes one hand at a time and sets up parameters
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Initialize cached action and state
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
        frame = cv2.flip(frame, 1)
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
            dist = MediaPipeHandController._euclidean_dist(thumb_tip, index_tip)

            #this is what maps the distance to the force
            norm = (dist - self._min_dist) / (self._max_dist - self._min_dist)
            norm = float(np.clip(norm, 0.0, 1.0))

            handed_label = "Right" #right and unless proven otherwise
            if results.multi_handedness:
                handed_label = results.multi_handedness[0].classification[0].label # get left or right hand label 

            direction = 1.0 if handed_label == "Right" else -1.0 # right hand pushes right, left hand pushes left
            signed_force = direction * norm * self._max_force
            self._cached_action[0] = signed_force / self._max_force # normalize to -1 to 1 for env action for boxpush
            self._last_force = signed_force # store for debug display
            display_frame = frame # for debug display

        if self._show_debug: # show the annotated frame with landmarks and force value
            annotated = display_frame.copy() # copy to draw on
            if results.multi_hand_landmarks: # if hand is detected, draw landmarks
                self._mp_drawing.draw_landmarks(
                    annotated, results.multi_hand_landmarks[0], self._mp_hands.HAND_CONNECTIONS
                )
            cv2.putText( # draw force value w contrasting outline for readability
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
            cv2.imshow(self._window_name, annotated) # show the annotated frame
            if cv2.waitKey(1) & 0xFF == 27: # if ESC is pressed, stop control
                self._active = False

        return self._cached_action

    def is_active(self) -> bool: # check for bool to see if control is active
        return self._active

    def close(self):
        self._hands.close()
        self._cap.release()
        if self._show_debug:
            cv2.destroyWindow(self._window_name)


class PolicyController:
    """Loads a trained behavior cloning policy and produces actions from observations."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        hidden_sizes = ckpt.get("hidden_sizes", [64, 64])
        obs_dim = ckpt.get("obs_dim")
        if obs_dim is None:
            obs_dim = ckpt["obs_mean"].shape[0]
        self.device = torch.device(device)
        self.model = BehaviorCloningPolicy(obs_dim, hidden_sizes).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.obs_mean = ckpt["obs_mean"].to(self.device)
        self.obs_std = ckpt["obs_std"].to(self.device)
        self.obs_std = torch.where(self.obs_std < 1e-6, torch.ones_like(self.obs_std), self.obs_std)
        self._cached_action = np.zeros(1, dtype=np.float32)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        obs_norm = (obs_tensor - self.obs_mean) / self.obs_std
        with torch.no_grad():
            action = self.model(obs_norm).cpu().numpy()
        self._cached_action[0] = action[0, 0]
        return self._cached_action


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
        show_debug=args.show_hand_debug, # apparently python doesnt like hyphens in variable names but this works fine
    )
    # Initialize demo storage
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


def run_policy(args) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    env = push(render_mode=args.render_mode)
    controller = PolicyController(args.policy_path)
    demos_obs: List[np.ndarray] = []
    demos_act: List[np.ndarray] = []

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            print(f"Episode {ep+1}/{args.episodes} - target={env.x_target:.3f}")

            while not done:
                action = controller.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                demos_obs.append(obs)
                demos_act.append(action.copy())
                obs = next_obs

            print("Episode end")
    finally:
        env.close()

    return demos_obs, demos_act


def save_demos(demos_obs: List[np.ndarray], demos_act: List[np.ndarray],mode:str):
    if not demos_obs or not demos_act:
        return
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    demos_obs_arr = np.stack(demos_obs)
    demos_act_arr = np.stack(demos_act)
    if mode == "policy":
        obs_filename = data_dir / "policy_obs.npy"
        act_filename = data_dir / "policy_act.npy"
    else:
        obs_filename = data_dir / "demos_obs.npy"
        act_filename = data_dir / "demos_act.npy"
    np.save(obs_filename, demos_obs_arr)
    np.save(act_filename, demos_act_arr)
    print(f"Saved {mode} data to {obs_filename} and {act_filename}")


def main():
    args = parse_args()

    if args.control == "mediapipe":
        demos_obs, demos_act = run_mediapipe(args)
    elif args.control == "policy":
        demos_obs, demos_act = run_policy(args)
    else:
        demos_obs, demos_act = run_keyboard(args)

    save_demos(demos_obs, demos_act,args.control)


if __name__ == "__main__":
    main()
