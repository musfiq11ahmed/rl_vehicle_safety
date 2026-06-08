import sys
sys.path.append(".")
from envs.vehicle_env import VehicleSafetyEnv

env = VehicleSafetyEnv()
obs, info = env.reset()

print("Environment created successfully!")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Initial observation:", obs)
print("  ego_speed  :", obs[0])
print("  lead_speed :", obs[1])
print("  gap        :", obs[2])
print("  ttc        :", obs[3])

# Run 5 random steps
print("\nRunning 5 random steps...")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1} | action: {action[0]:.2f} | reward: {reward:.2f} | ttc: {info['ttc']:.2f}s | gap: {info['gap']:.2f}m")
    if terminated:
        print("Collision occurred — episode ended")
        break

print("\nEnvironment test passed!")