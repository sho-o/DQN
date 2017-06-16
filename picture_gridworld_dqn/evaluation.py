import numpy as np
import environment
import copy

class Evaluation():
	def __init__(self, comment, test_pics, s_init, actions, max_step, reward_clip, test_iter):
		self.env = environment.Environment(test_pics)
		self.s_init = s_init
		self.actions = actions
		self.comment = comment
		self.max_step = max_step
		self.reward_clip = reward_clip
		self.test_iter = test_iter
		f = open("result/{}/evaluation/evaluation.csv".format(comment), "a")
		f.write("episode,total_step,reward_mean,reward_std,step_mean,step_std,success_times,success_step_mean\n")
		f.close()

	def __call__(self, agt, learning_episode, learning_total_step):
		reward_list = []
		step_num_list = []
		success_counter = 0
		success_step_num_list = []
		for episode in range(self.test_iter):
			self.env.make_episode_pics()
			episode_reward = 0
			s = self.s_init
			pic_s = self.env.s_to_pic(s)

			for steps in range(self.max_step):
				a, _ = agt.policy(pic_s, eva=True)
				new_s = self.env.generate_next_s(s, self.actions[a])
				pic_new_s = self.env.s_to_pic(new_s)
				r = self.env.make_reward(s, self.actions[a], self.reward_clip)
				done = self.env.judge_finish(new_s)
				episode_reward += r
				if done:
					break
				#prepare next s
				s = new_s[:]
				pic_s = np.array(pic_new_s)

			reward_list.append(episode_reward)
			step_num_list.append(steps+1)
			if episode_reward == 1:
				success_counter += 1
				success_step_num_list.append(steps+1)

		reward_array = np.array(reward_list)
		step_array = np.array(step_num_list)
		reward_mean = np.average(reward_array)
		reward_std = np.std(reward_array)
		step_mean = np.average(step_array)
		step_std = np.std(step_array)
		success_times = success_counter
		if len(success_step_num_list) > 0:
			success_step_mean = float(sum(success_step_num_list))/len(success_step_num_list)
		else:
			success_step_mean = 0.0
		self.make_eval_log(self.comment, learning_episode, learning_total_step, reward_mean, reward_std, step_mean, step_std, success_times, success_step_mean)

	def make_eval_log(self, comment, learning_episode, learning_total_step, reward_mean, reward_std, step_mean, step_std, success_times, success_step_mean):
		f = open("result/{}/evaluation/evaluation.csv".format(comment), "a")
		f.write(str(learning_episode+1) + "," + str(learning_total_step) + "," + str(reward_mean) + "," + str(reward_std)+ "," + str(step_mean) + "," + str(step_std)+ "," + str(success_times) + "," + str(success_step_mean) + "\n")
		f.close()


