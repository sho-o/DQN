import numpy as np
import agent
import environment
import copy

class Evaluation():
	def __init__(self, comment, pic_kind, s_init, actions, max_step):
		self.env = environment.Environment(pic_kind)
		self.s_init = s_init
		self.actions = actions
		self.comment = comment
		self.max_step = max_step
		f = open("result/{}/evaluation/evaluation.csv".format(comment), "a")
		f.write("episode,total_step,reward,episode_step\n")
		f.close()

	def __call__(self, agt, learning_episode, learning_total_step):
		for episode in range(1):
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

		self.make_eval_log(self.comment, learning_episode, learning_total_step, episode_reward, steps)

	def make_eval_log(self, comment, learning_episode, learning_total_step, episode_reward, steps):
		f = open("result/{}/evaluation/evaluation.csv".format(comment), "a")
		f.write(str(learning_episode+1) + "," + str(learning_total_step) + "," + str(episode_reward) + "," + str(steps+1) + "\n")
		f.close()


