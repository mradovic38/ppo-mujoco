import gymnasium as gym

env = gym.make("HalfCheetah-v5")

obs, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

print("MuJoCo works.")
env.close()