import numpy as np

Fmax = 10.0

# Human teleop
human_obs = np.load("demos_obs.npy")  # shape (timesteps, obs_dim)
human_act = np.load("demos_act.npy")  # shape (timesteps, 1)

# Policy
policy_obs = np.load("policy_obs.npy")
policy_act = np.load("policy_act.npy")

force_human = human_act * Fmax  # shape same as act
force_policy = policy_act * Fmax  # shape same as act



# Extract position (x) and target relative position (rel)
human_x = human_obs[:, 0]
human_rel = human_obs[:, 2]
policy_x = policy_obs[:, 0]
policy_rel = policy_obs[:, 2]

# # Final distance to target
# human_final_err = abs(human_rel[-1])
# policy_final_err = abs(policy_rel[-1])
# print("Final error human:", human_final_err)
# print("Final error policy:", policy_final_err)

# Mean absolute error over all timesteps
human_mae = np.mean(np.abs(human_rel))
policy_mae = np.mean(np.abs(policy_rel))
print("Mean absolute error human:", human_mae)
print("Mean absolute error policy:", policy_mae)

import matplotlib.pyplot as plt

dt = 0.01  # adjust to your environment timestep
episodes = 8  # same as used in teleop.py

def compute_episode_work(obs, act, Fmax, dt, episodes):
    steps_per_ep = len(obs) // episodes
    ep_work = []
    for ep in range(episodes):
        start = ep * steps_per_ep
        end = start + steps_per_ep
        force = act[start:end].flatten() * Fmax
        velocity = obs[start:end, 1]  # v from obs
        power = force * velocity
        work = np.sum(power * dt)
        ep_work.append(work)
    return np.array(ep_work)

work_human_ep = compute_episode_work(human_obs, human_act, Fmax, dt, episodes)
work_policy_ep = compute_episode_work(policy_obs, policy_act, Fmax, dt, episodes)

# Average work per episode
avg_human_work = np.mean(work_human_ep)
avg_policy_work = np.mean(work_policy_ep)

print(f"Average work per episode - Human: {avg_human_work:.2f} J, Policy: {avg_policy_work:.2f} J")

def final_distance(obs, episodes):
    steps_per_ep = len(obs) // episodes
    final_dist = []
    for ep in range(episodes):
        last_obs = obs[(ep+1)*steps_per_ep - 1]
        final_dist.append(abs(last_obs[2]))  # rel distance to target
    return np.array(final_dist)

human_final_dist = final_distance(human_obs, episodes)
policy_final_dist = final_distance(policy_obs, episodes)
print(f"Human final distance: {np.mean(human_final_dist):.3f} ± {np.std(human_final_dist):.3f}")
print(f"Policy final distance: {np.mean(policy_final_dist):.3f} ± {np.std(policy_final_dist):.3f}")

# WORK PER EPISODE COMPARISON
plt.figure(figsize=(6,4))
plt.bar(range(episodes), work_human_ep, alpha=0.6, label="Human")
plt.bar(range(episodes), work_policy_ep, alpha=0.6, label="Policy")
plt.xlabel("Episode")
plt.ylabel("Total Work (Joules)")
plt.title("Total Work per Episode")
plt.legend()
plt.show()

# FINAL DISTANCE COMPARISON PER EPISODE
plt.figure(figsize=(6,4))
plt.bar(range(episodes), human_final_dist, alpha=0.6, label="Human")
plt.bar(range(episodes), policy_final_dist, alpha=0.6, label="Policy")
plt.xlabel("Episode")
plt.ylabel("Final Distance to Target")
plt.title("Final Distance to Target per Episode")
plt.legend()
plt.show()



plt.figure(figsize=(8,4))
plt.plot(human_x, label="Human Position", alpha=0.7)
plt.plot(policy_x, label="Policy Position", alpha=0.7)
plt.xlabel("Timestep")
plt.ylabel("Block X Position")
plt.title("Block position: Human vs Policy")
plt.legend()
plt.grid(True)
plt.show()

# Optionally, plot relative distance to target
plt.figure(figsize=(8,4))
plt.plot(human_rel, label="Human Distance to Target")
plt.plot(policy_rel, label="Policy Distance to Target")
plt.xlabel("Timestep")
plt.ylabel("Distance to Target")
plt.title("Distance to target: Human vs Policy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(human_act, label="Human Action", alpha=0.7)
plt.plot(policy_act, label="Policy Action", alpha=0.7)
plt.xlabel("Timestep")
plt.ylabel("Normalized Force")
plt.title("Actions: Human vs Policy")
plt.legend()
plt.grid(True)
plt.show()

