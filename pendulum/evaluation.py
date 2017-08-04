import numpy as np
import copy
import gym
from gym import wrappers

class Evaluation():
	def __init__(self, directory_path, game, name, comment, max_step, skip_size, eval_iter, max_initial_noop):
		self.game = game
		if game == "doom":
			import ppaquette_gym_doom
		self.env = gym.make(name)
		self.comment = comment
		self.max_step = max_step
		self.skip_size = skip_size
		self.eval_iter = eval_iter
		self.max_initial_noop = max_initial_noop
		self.directory_path = directory_path
		f = open("{}/{}/evaluation/evaluation.csv".format(self.directory_path, comment), "a")
		f.write("episode,total_step,reward_mean,reward_std,step_mean,step_std\n")
		f.close()

	def __call__(self, agt, pre, learning_episode, learning_total_step):
		reward_list = []
		step_num_list = []
		initial_noop = 0
		pendulum_actions = [-1,-0.1, -0.01, 0, 0.01, 0.1, 1]
		for episode in range(self.eval_iter):
			episode_reward = 0
			if self.max_initial_noop > 0:
				initial_noop = np.random.randint(self.max_initial_noop)
			s = self.env.reset()

			for steps in range(self.max_step):
				a, value = agt.policy(s, eva=True)
				if steps < initial_noop:
					a = 0
				if self.game == "doom":
					action = pre.action_convert(a)
					obs, r, done, info = self.env.step(action)
				if self.game == "atari":
					obs, r, done, info = self.env.step(a)
				if self.game == "pendulum":
					new_s, r, done, info = self.env.step([pendulum_actions[a]])
				#obs_processed = pre.one(obs)

				#new_s = np.asanyarray([s[1], s[2], s[3], obs_processed], dtype=np.uint8)
				#r_clipped = pre.reward_clip(r)

				episode_reward += r

				if done:
					break
				#prepare next s
				s = np.array(new_s)

			reward_list.append(episode_reward)
			step_num_list.append(steps+1)

		reward_array = np.array(reward_list)
		step_array = np.array(step_num_list)
		reward_mean = np.average(reward_array)
		reward_std = np.std(reward_array)
		step_mean = np.average(step_array)
		step_std = np.std(step_array)
		self.make_eval_log(self.comment, learning_episode, learning_total_step, reward_mean, reward_std, step_mean, step_std)

	def make_eval_log(self, comment, learning_episode, learning_total_step, reward_mean, reward_std, step_mean, step_std):
		f = open("{}/{}/evaluation/evaluation.csv".format(self.directory_path, comment), "a")
		f.write(str(learning_episode+1) + "," + str(learning_total_step) + "," + str(reward_mean) + "," + str(reward_std)+ "," + str(step_mean) + "," + str(step_std) + "\n")
		f.close()


