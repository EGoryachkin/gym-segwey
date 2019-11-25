import gym

env = gym.make('gym_segwey:segwey-v0')
x = env.reset()
env.render()

for _ in range(1000):
  env.render()
  action = env.tet.dot(x)
  x, reward, done, info = env.step(action)
  print(x[1], done)
  if done:
    x = env.reset()
env.close()