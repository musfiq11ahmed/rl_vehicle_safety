import gymnasium as gym
import highway_env

env = gym.make("highway-v0", config={
    "action": {
        "type": "ContinuousAction"
    }
})

obs, info = env.reset()
print("Continuous HighwayEnv configured successfully!")
print("Observation shape:", obs.shape)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Action space shape:", env.action_space.shape)
print("\nInitial observation:\n", obs)