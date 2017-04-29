import numpy as np
import scipy.misc as spm

class Preprocess():

	def reward_clip(self, r):
		if r > 0: r = 1.0
		if r < 0: r = -1.0
		return r

	def gray(self, obs):
		obs_gray = 0.299*obs[:, :, 0] + 0.587*obs[:, :, 1] + 0.114*obs[:, :, 2]
		return obs_gray

	def max(self, obs1, obs2):
		obs_max = np.maximum(obs1, obs2)
		return obs_max

	def downscale(self, obs):
		obs_down = spm.imresize(obs, (84, 84))
		return obs_down

	def one(self, obs):
		processed = self.downscale(self.gray(obs))
		return processed

	def two(self, obs1, obs2):
		processed = self.downscale(self.gray(self.max(obs1, obs2)))
		return processed

	def action_convert(self, a):
		action = [0] * 43
		if a == 1:
			action[0] = 1
		if a == 2:
			action[14] = 1
		if a == 3:
			action[15] = 1


		return action