import mediapipe
import cv2
import mujoco
import mujoco.viewer
import multiprocessing
import gymnasium as gym
import pygame
from gymnasium import spaces
import numpy as np
from pathlib import Path


xml_path = "boxpush.xml"

class push(gym.Env):
    metadata = {"render.modes": ["human", "none"]}

    def __init__(self, xml_path=xml_path, render_mode="human"):
        super().__init__()

        #necessary mujoco components
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Initialize environment components here

        self.dt = self.model.opt.timestep

        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slider_x")
        self.body_target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.actuator_id = 0

        #action space
        self.Fmax = 10.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.max_steps = 200
        self.step_count = 0

        self.target_min = -0.8
        self.target_max = 0.8

        self.render_mode = render_mode
        self.viewer = None
        
        if self.render_mode == "human":
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception:
                # Fallback: use offline rendering with pixel data
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
                print("Using offline rendering (headless mode with frame output)")

    def _get_block_state(self):
        """Get block position and velocity from mujoco."""
        x = float(self.data.qpos[self.joint_id])
        v = float(self.data.qvel[self.joint_id])
        return x, v

    def _set_block_state(self, x, v):
        """Set block position and velocity in mujoco."""
        self.data.qpos[self.joint_id] = x
        self.data.qvel[self.joint_id] = v
        mujoco.mj_forward(self.model, self.data)

    def _set_target_pos(self, pos):
        """Set target position by modifying the target body position via XML or kinematic forward pass."""

        pass

    def _get_obs(self):
        """Get observation: [block_x, block_v, relative_pos_to_target]."""
        x, v = self._get_block_state()
        rel = self.x_target - x
        obs = np.array([x, v, rel], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        mujoco.mj_resetData(self.model, self.data)


        #========RANDOM TARGET AND INITIAL STATE=========
        x0 = self.np_random.uniform(-0.5, 0.5)
        v0 = 0.0
        self._set_block_state(x0, v0)

        #sample random target position
        self.x_target = self.np_random.uniform(self.target_min, self.target_max)
        self._set_target_pos(self.x_target)
        #===============================================
        self.step_count = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        #clip and scale action
        a = float(np.clip(action[0], -1.0, 1.0))
        F = a * self.Fmax
        self.data.ctrl[self.actuator_id] = F

        #advance simulation
        mujoco.mj_step(self.model, self.data)
        
        #build reward
        obs = self._get_obs()
        x, v = self._get_block_state()
        x_err = x - self.x_target

        reward = -abs(x_err) - 0.001 * (F ** 2)

        epsilon = 0.02  # success radius
        terminated = (abs(x_err) < epsilon)
        truncated = (self.step_count >= self.max_steps)

        info = {}

        #   optionally render
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment using MuJoCo viewer or offline renderer."""
        if self.render_mode == "none":
            return
        
        if self.viewer is not None:
            # Live viewer is available
            self.viewer.sync()
        else:
      
            try:
                self.renderer.update_scene(self.data)
                img = self.renderer.render()
               
                cv2.imshow("MuJoCo Simulation", img)
                cv2.waitKey(1) 
            except Exception:
                pass 

    def close(self):
       
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass



