import numpy as np
import copy
import gym
import ppaquette_gym_doom

class Evaluation():
	def __init__(self, game, name, comment, max_step, skip_mode, skip_size):
		self.game = game
		self.env = gym.make(name)
		self.comment = comment
		self.max_step = max_step
		self.skip_mode = skip_mode
		self.skip_size = skip_size
		f = open("result/{}/evaluation/evaluation.csv".format(comment), "a")
		f.write("episode,total_step,reward_mean,reward_std,step_mean,step_std\n")
		f.close()

	def __call__(self, agt, pre, learning_episode, learning_total_step):
		reward_list = []
		step_num_list = []
		for episode in range(100):
			episode_reward = 0
			obs = self.env.reset()
			s = np.zeros((4, 84, 84), dtype=np.uint8)
			s[3] = pre.one(obs)

			for steps in range(self.max_step):

				if self.game == "atari" and self.skip_mode == "deterministic":
					r = 0
					a, value = agt.policy(s, eva=True)
					action = self.env._action_set[a]
					for i in range(self.skip_size):
						r += self.env.ale.act(action)
						obs_prev = np.array(obs)
						obs = self.env._get_obs()
					done = self.env.ale.game_over()
					obs_processed = pre.two(obs_prev, obs)

				else:
					a, value = agt.policy(s, eva=True)
					if self.game == "doom":
						action = pre.action_convert(a)
						obs, r, done, info = self.env.step(action)
					if self.game == "atari":
						obs, r, done, info = self.env.step(a)
					obs_processed = pre.one(obs)

				new_s = np.asanyarray([s[1], s[2], s[3], obs_processed], dtype=np.uint8)
				r_clipped = pre.reward_clip(r)

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
		f = open("result/{}/evaluation/evaluation.csv".format(comment), "a")
		f.write(str(learning_episode+1) + "," + str(learning_total_step) + "," + str(reward_mean) + "," + str(reward_std)+ "," + str(step_mean) + "," + str(step_std) + "\n")
		f.close()


