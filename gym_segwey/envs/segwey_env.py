import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random

# x = [x, phi, dx, dphi]

class SegweyEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		
		self.Ak = np.array([[1.0000, -0.0001,  0.0038, 0.0001],\
			[0.0000,  1.0021,  0.1262, 0.0074],\
			[0.0000, -0.0085,  0.2971, 0.0147],\
			[0.0000,  0.2614, 14.2212, 0.7036] ])
		self.Bk = np.array([[ 0.0001],\
			[-0.0015],\
			[ 0.0123],\
			[-0.2483] ])
		self.tet = np.array([0., 28.5536, 155.3009, 7.7108])

		
		self.length = 0.5 # actually half the pole's length
		self.max_u = 4
		self.x_threshold = 2.4
		self.theta_threshold_radians = 100.6
		self.h = 0.01
		self.state = None

		self.steps_beyond_done = None
		self.viewer = None
	# C = np.array([0., 1., 0., 0.])

	def step(self, action):

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
			# Pole just fell!
			self.steps_beyond_done = 0
			reward = 1.0
		else:
			if self.steps_beyond_done == 0:
				logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
			reward = 0.0
		
		return np.array(self.state), reward, done, {}
		
	def reset(self):
		self.state = np.array([[0.],[random.gauss(0, 1.5)],[0.],[0.]])
		# self.state = np.array([[0.],[random.gauss(0, 0.5)],[0.],[0.]])
		self.steps_beyond_done = None
		return np.array(self.state)

		
	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400

		world_width = self.x_threshold*2
		scale = screen_width/world_width
		carty = 100 # TOP OF CART
		polewidth = 10.0
		polelen = scale * (2 * self.length)
		cartwidth = 50.0
		cartheight = 30.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			axleoffset =cartheight/4.0
			cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
			pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			pole.set_color(.8,.6,.4)
			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)
			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth/2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5,.5,.8)
			self.viewer.add_geom(self.axle)
			self.track = rendering.Line((0,carty), (screen_width,carty))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

			self._pole_geom = pole

		if self.state is None: return None

		# Edit the pole polygon vertex
		pole = self._pole_geom
		l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
		pole.v = [(l,b), (l,t), (r,t), (r,b)]

		x = self.state
		cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(-x[2])

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
