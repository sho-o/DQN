import numpy as np
import environment
import copy

class Evaluation():
	def __init__(self, comment, pic_kind, s_init, actions, max_step):
		self.env = environment.Environment(pic_kind, 1)
		self.s_init = s_init
		self.actions = actions
		self.comment = comment
		self.max_step = max_step
		f = open("result/{}/evaluation/evaluation.csv".format(comment), "a")
		f.write("episode,total_step,reward_mean,reward_std,step_mean,step_std\n")
		f.close()

	def __call__(self, agt, learning_episode, learning_total_step):
		reward_list = []
		step_num_list = []
		for episode in range(30):
			self.env.make_episode_pics()
			episode_reward = 0
			s = self.s_init
			pic_s = self.env.s_to_pic(s)

			for steps in range(self.max_step):
				a, _ = agt.policy(pic_s, eva=True)
				new_s = self.env.generate_next_s(s, self.actions[a])
				pic_new_s = self.env.s_to_pic(new_s)
				r = self.env.make_reward(s, self.actions[a])
				done = self.env.judge_finish(new_s)
				episode_reward += r
				if done:
					break
				#prepare next s
				s = new_s[:]
				pic_s = np.array(pic_new_s)

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


