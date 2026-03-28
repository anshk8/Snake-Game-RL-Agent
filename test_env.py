from agent.snake_env import SnakeEnv

env = SnakeEnv()
obs, _ = env.reset()

episode = 1
episode_score = 0
episode_steps = 0
scores = []

for i in range(5000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_score += reward
    episode_steps += 1

    if terminated or truncated:
        scores.append(info['score'])
        print(f"Episode {episode}: steps={episode_steps} score={info['score']} total_reward={episode_score:.1f}")
        obs, _ = env.reset()
        episode += 1
        episode_score = 0
        episode_steps = 0

print(f"\nAverage score over {len(scores)} episodes: {sum(scores)/len(scores):.2f}")