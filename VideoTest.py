import gym
from gym.wrappers import Monitor
env = gym.wrappers.Monitor(gym.make('CartPole-v0'), './vid', video_callable=lambda episode_id: True, resume=True)

state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    state_next, reward, done, info = env.step(action)
env.close()
