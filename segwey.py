import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random

# x = [x, phi, dx, dphi]

class segwey(object):
	def __init__(self):
		
		self.h = 0.05
		self.Ak = np.array([
			[1., -0.0003, 0.0104, 0.0008],
			[0.,  1.0140, 0.4316, 0.0412],
			[0., -0.0038, 0.1828, 0.0168],
			[0.,  0.4514, 8.9706, 0.8257]])
		self.Bk = np.array([0.0005, -0.0051, 0.0326, -0.3551])
		self.tet = np.array([0., 44.1406, 147.5833, 13.6393])
		# self.h = 0.01
		# self.Ak = np.array([[1.0000, -0.0001,  0.0038, 0.0001],\
		# 	[0.0000,  1.0021,  0.1262, 0.0074],\
		# 	[0.0000, -0.0085,  0.2971, 0.0147],\
		# 	[0.0000,  0.2614, 14.2212, 0.7036] ])
		# self.Bk = np.array([[ 0.0001],\
		# 	[-0.0015],\
		# 	[ 0.0123],\
		# 	[-0.2483] ])
		# # C = np.array([0., 1., 0., 0.])
		# self.Bk = self.Bk.reshape((4,))
		# self.tet = np.array([0., 28.5536, 155.3009, 7.7108])
		
		self.discrete = False
		self.discrete_level = 25
		self.n_actions = 1 + 2*self.discrete_level

		self.max_u = 200
		self.state_dim = (4,)
		self.x_threshold = 2.4
		self.theta_threshold_radians = 2.0
		self.state = None

		self.steps_beyond_done = None

	def set_param(self, discrete_level=25, max_u = 200, discrete = True):
		self.discrete = discrete
		self.discrete_level = discrete_level
		self.n_actions = 1 + 2*self.discrete_level
		self.max_u = max_u	

	def step(self, action):
		if self.discrete:
			action = (action-self.discrete_level)/self.discrete_level*self.max_u
		x = self.Ak.dot(self.state)+self.Bk*action
		self.state = x

		done =  x[0] < -self.x_threshold \
			 or x[0] > self.x_threshold \
			 or x[1] < -self.theta_threshold_radians \
			 or x[1] > self.theta_threshold_radians
		done = bool(done)

		if not done:
			reward = 1.0
		elif self.steps_beyond_done is None:
			self.steps_beyond_done = 0
			reward = 1.0
		else:
			if self.steps_beyond_done == 0:
				logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
			reward = 0.0
		
		return np.array(self.state).reshape((4,)), reward, done, {}
		
	def reset(self):
		self.state = np.array([[0.],[random.gauss(0, 0.5)],[0.],[0.]]).reshape((4,))
		self.steps_beyond_done = None
		return np.array(self.state).reshape((4,))
