# teleop_collect.py
import pygame
import numpy as np
from boxpush import push

def get_action_from_keys():
    keys = pygame.key.get_pressed()
    a = 0.0
    if keys[pygame.K_LEFT]:
        a = -1.0
    elif keys[pygame.K_RIGHT]:
        a = 1.0
    else:
        a = 0.0
    return np.array([a], dtype=np.float32)

def main():
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("1D Push Teleop - Use arrow keys to control")

    clock = pygame.time.Clock()


    env = push(render_mode="human")
    demos_obs = []
    demos_act = []

    num_episodes = 20

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        print(f"Episode {ep+1}/{num_episodes} - target={env.x_target:.3f}")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            action = get_action_from_keys()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

     
            demos_obs.append(obs)
            demos_act.append(action)

            obs = next_obs

        
            screen.fill((0, 0, 0))
            pygame.display.flip()
            clock.tick(50)  # ~50 Hz

        print("Episode end")

    env.close()
    pygame.quit()

    # Convert to numpy and save
    demos_obs_arr = np.stack(demos_obs)
    demos_act_arr = np.stack(demos_act)

    np.save("demos_obs.npy", demos_obs_arr)
    np.save("demos_act.npy", demos_act_arr)
    print("Saved demos to demos_obs.npy and demos_act.npy")

if __name__ == "__main__":
    main()
