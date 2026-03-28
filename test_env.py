from agent.snake_env import SnakeEnv
from utils.constants import WIDTH, HEIGHT, GRID_SIZE, CELL_SIZE

def test1():
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

def testHumanMode():
    from agent.snake_env import SnakeEnv
    import time

    env = SnakeEnv(render_mode="human")
    obs, _ = env.reset()

    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    testHumanMode()