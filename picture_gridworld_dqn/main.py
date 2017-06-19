import numpy as np
import agent
import environment
from chainer import cuda, serializers
import argparse
import time
import os
import sys
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pylab as plt
import evaluation
import loss_loger
import pandas as pd
from sklearn.datasets import fetch_mldata


parser = argparse.ArgumentParser()
parser.add_argument('--comment', '-c', default='', type=str, help='comment to distinguish output')
parser.add_argument('--gpu', '-g', default= -1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--directory_path', '-dp', default="result", type=str, help='directory path')
parser.add_argument('--pic_kind', '-k', default="mnist", type=str, help='kind of pictures')
parser.add_argument('--exp_policy', '-p', default="epsilon_greedy", type=str, help='explorlation policy')
parser.add_argument('--epsilon_decrease_end', '-ee', default=10**4, type=int, help='the step number of the end of epsilon decrease')
parser.add_argument('--max_episode', '-e', default=10**7, type=int, help='number of episode to learn')
parser.add_argument('--max_step', '-s', default=1000, type=int, help='max steps per episode')
parser.add_argument('--finish_step', '-fs', default=10**6, type=int, help='end of the learning')
parser.add_argument('--q_update_freq', '-q', default=4, type=int, help='q update freaquency')
parser.add_argument('--fixed_q_update_freq', '-f', default=10**4, type=int, help='fixed q update frequency')
parser.add_argument('--save_freq', '-sf', default=5*10**4, type=int, help='save frequency')
parser.add_argument('--eval_freq', '-ef', default=10**4, type=int, help='evaluatuin frequency')
parser.add_argument('--print_freq', '-pf', default=1, type=int, help='print result frequency')
parser.add_argument('--graph_freq', '-gf', default=10**5, type=int, help='make graph frequency')
parser.add_argument('--initial_exploration', '-i', default=10**3, type=int, help='number of initial exploration')
parser.add_argument('--batch_size', '-b', type=int, default=32, help='learning minibatch size')
parser.add_argument('--memory_size', '-ms', type=int, default=10**6, help='replay memory size')
parser.add_argument('--input_slides', '-is', type=int, default=1, help='number of input slides')
parser.add_argument('--net_type', '-n', type=str, default="full", help='network type (full, conv, DQN)')
parser.add_argument('--pic_size', '-ps', type=int, default=28, help='nput pic size')
parser.add_argument('--discount', '-d', type=float, default=0.99, help='discount factor')
parser.add_argument('--rms_eps', '-re', type=float, default=0.01, help='RMSProp_epsilon')
parser.add_argument('--rms_lr', '-lr', type=float, default=0.00025, help='RMSProp_learning_rate')
parser.add_argument('--optimizer_type', '-o', type=str, default="rmsprop", help='type of optimizer')
parser.add_argument('--start_point', '-sp', type=int, default=1, help='start point')
parser.add_argument('--mode', '-m', type=str, default="default", help='default or regularize or mix')
parser.add_argument('--threshold', '-t', type=float , default=0.001, help='regularization threshold')
parser.add_argument('--penalty_weight', '-pw', type=float, default=1.0, help='regularization penalty weight')
parser.add_argument('--mix_rate', '-mr', type=float, default=0, help='target_mix _rate')
parser.add_argument('--training_size', '-ts', type=int, default=2000, help='number of kinds of training pictures')
parser.add_argument('--test_size', '-tes', type=int, default=2000, help='number of kinds of test pictures')
parser.add_argument('--loss_log_iter', '-li', type=int, default=10, help='(batch) iteration  compute average loss and penalty (1batch=32)')
parser.add_argument('--loss_log_freq', '-lf', default=20, type=int, help='record loss frequency per fixed q upddate')
parser.add_argument('--rolling_mean_width', '-r', default=1000, type=int, help='width of rolling mean')
parser.add_argument('--reward_clip', '-rc', default=1, type=int, help='clip the reward or not')
parser.add_argument('--test_iter', '-ti', type=int, default=100, help='test iteration times')
args = parser.parse_args()

def run(args):
	comment = args.comment
	gpu = args.gpu
	directory_path = args.directory_path
	pic_kind = args.pic_kind
	exp_policy = args.exp_policy
	epsilon_decrease_end = args.epsilon_decrease_end
	max_episode = args.max_episode
	max_step = args.max_step
	finish_step = args.finish_step
	q_update_freq = args.q_update_freq
	fixed_q_update_freq = args.fixed_q_update_freq
	save_freq = args.save_freq
	eval_freq = args.eval_freq
	print_freq = args.print_freq
	graph_freq = args.graph_freq
	initial_exploration = args.initial_exploration
	batch_size = args.batch_size
	memory_size = args.memory_size
	input_slides = args.input_slides
	net_type = args.net_type
	pic_size = args.pic_size
	discount = args.discount
	rms_eps = args.rms_eps
	rms_lr = args.rms_lr
	optimizer_type = args.optimizer_type
	start_point = args.start_point
	mode = args.mode
	threshold = args.threshold
	mix_rate = args.mix_rate
	penalty_weight = args.penalty_weight
	training_size = args.training_size
	test_size = args.test_size
	loss_log_iter = args.loss_log_iter
	loss_log_freq = args.loss_log_freq
	rolling_mean_width = args.rolling_mean_width
	reward_clip = args.reward_clip
	test_iter = args.test_iter
	s_init = [(start_point-1)%3, (start_point-1)/3]
	epsilon_decrease_wide = 0.9/(epsilon_decrease_end - initial_exploration)

	run_start = time.time()
	make_directries(directory_path, comment, ["network", "log", "evaluation", "loss"])
	if gpu >= 0:
		cuda.get_device(gpu).use()
	training_pics, test_pics = separate_data(pic_kind, training_size, test_size)
	print training_pics.shape
	print test_pics.shape
	env = environment.Environment(training_pics)
	actions = ["up", "down", "right", "left"] 
	num_of_actions = len(actions)
	agt = agent.Agent(exp_policy, net_type, gpu, pic_size, num_of_actions, memory_size, input_slides, batch_size, discount, rms_eps, rms_lr, optimizer_type, mode, threshold, penalty_weight, mix_rate)
	eva = evaluation.Evaluation(directory_path, comment, test_pics, s_init, actions, max_step, reward_clip, test_iter)
	loss_log = loss_loger.Loss_Log(directory_path, comment, loss_log_iter, gpu)
	total_step = 0
	fixed_q_update_counter = 0

	for episode in range(max_episode):
		episode_start = time.time()
		env.make_episode_pics()
		episode_reward = 0
		episode_value = 0
		s = s_init
		pic_s = env.s_to_pic(s)
		if total_step > finish_step:
			break

		for steps in range(max_step):
			a, value = agt.policy(pic_s)
			new_s = env.generate_next_s(s, actions[a])
			pic_new_s = env.s_to_pic(new_s)
			r = env.make_reward(s, actions[a], reward_clip)
			done = env.judge_finish(new_s)
			agt.store_experience(total_step, pic_s, a, r, pic_new_s, done)
			episode_reward += r
			episode_value += value
			total_step += 1

			#update and save
			if total_step > initial_exploration:
				if total_step % q_update_freq == 0:
					agt.q_update(total_step)
				if fixed_q_update_counter % loss_log_freq == 0 and fixed_q_update_counter > 0:
					loss_log(fixed_q_update_counter, total_step, agt)
				if total_step % fixed_q_update_freq == 0:
					print "----------------------- fixed Q update ------------------------------"
					agt.fixed_q_updqte()
					fixed_q_update_counter += 1
					if fixed_q_update_counter % loss_log_freq == 0:
						make_loss_log_file(directory_path, comment, fixed_q_update_counter)
					if fixed_q_update_counter % loss_log_freq == 1 and fixed_q_update_counter > 1:
							print "----------------------- make_loss_graph ------------------------------"
							make_loss_graph(directory_path, comment, fixed_q_update_counter-1)
				#if total_step % save_freq == 0:
					#print "----------------------- save the_model ------------------------------"
					#serializers.save_npz('{}/{}/network/q_{}.net'.format(directory_path, comment, total_step), agt.q)
				if total_step % eval_freq == 0:
					print "----------------------- evaluate the model ------------------------------"
					eva(agt, episode, total_step)
				if total_step % graph_freq == 0:
					print "----------------------- make graph ------------------------------"
					make_test_graph(directory_path, comment)
					make_training_graph(directory_path, comment, rolling_mean_width)
				agt.epsilon = max(0.1, agt.epsilon - epsilon_decrease_wide)

			#log, print_result
			if done:
				run_time = time.time() - run_start
				episode_time = time.time() - episode_start
				episode_average_value = episode_value/steps
				make_log(directory_path, comment, episode, episode_reward, episode_average_value, agt.epsilon, steps, total_step, run_time)
				if total_step % print_freq == 0:
					print_result(episode, steps, episode_reward, episode_time, agt.epsilon, total_step, run_time)
				break

			#prepare next s
			s = new_s[:]
			pic_s = np.array(pic_new_s)

def make_directries(directory_path, comment, dirs):
	for d in dirs:
		if not os.path.exists("{}/{}/".format(directory_path, comment) + d):
			os.makedirs("{}/{}/".format(directory_path, comment) + d)

def make_loss_log_file(directory_path, comment, fixed_q_update_counter):
	f = open("{}/{}/loss/{}_loss.csv".format(directory_path, comment, fixed_q_update_counter), "a")
	f.write("fixed_q_update_counter,total_step,loss_mean,loss_std,penalty_mean,penalty_std\n")
	f.close()

def make_log(directory_path, comment, episode, episode_reward, episode_average_value, epsilon, steps, total_step, run_time):
	f = open("{}/{}/log/log.csv".format(directory_path, comment), "a")
	if episode == 0:
		f.write("episode,reward,average_value,epsilon,episode_step,total_step,run_time\n")
	f.write(str(episode+1) + "," + str(episode_reward) + "," + str(episode_average_value) + "," + str(epsilon) + ',' + str(steps+1) + ',' + str(total_step) + ',' + str(run_time) + "\n")
	f.close()

def make_test_graph(directory_path, comment):
	df = pd.read_csv("{}/{}/evaluation/evaluation.csv".format(directory_path, comment))
	total_step = np.array(df.loc[:, "total_step"].values, dtype=np.int)
	reward_mean = np.array(df.loc[:, "reward_mean"].values, dtype=np.float)
	success_times = np.array(df.loc[:, "success_times"].values, dtype=np.int)
	success_step_mean = np.array(df.loc[:, "success_step_mean"].values, dtype=np.float)
	#reward_std = np.array(df.loc[:, "reward_std"].values, dtype=np.float)
	step_mean = np.array(df.loc[:, "step_mean"].values, dtype=np.float)
	#step_std = np.array(df.loc[:, "step_std"].values, dtype=np.float)
	plt.figure()
	plt.plot(total_step, reward_mean, color="r")
	#plt.fill_between(total_step, reward_mean+reward_std, reward_mean-reward_std, facecolor='red', alpha=0.3)
	plt.savefig("{}/{}/evaluation/reward.png".format(directory_path, comment))
	plt.figure()
	plt.plot(total_step, step_mean, color="b")
	#plt.fill_between(total_step, step_mean+step_std, step_mean-step_std, facecolor='blue', alpha=0.3)
	plt.savefig("{}/{}/evaluation/step.png".format(directory_path, comment))
	plt.figure()
	plt.plot(total_step, success_times, color="g")
	plt.savefig("{}/{}/evaluation/success_times.png".format(directory_path, comment))
	plt.figure()
	plt.plot(total_step, success_step_mean, color="c")
	plt.savefig("{}/{}/evaluation/success_step_mean.png".format(directory_path, comment))

def make_training_graph(directory_path, comment, rolling_mean_width):
	df = pd.read_csv("{}/{}/log/log.csv".format(directory_path, comment))
	total_step = np.array(df.loc[:, "total_step"].values, dtype=np.int)
	reward = np.array(df.loc[:, "reward"].values, dtype=np.int)
	reward = pd.Series(reward).rolling(window=rolling_mean_width).mean()
	episode_step = np.array(df.loc[:, "episode_step"].values, dtype=np.int)
	episode_step = pd.Series(episode_step).rolling(window=rolling_mean_width).mean()
	plt.figure()
	plt.plot(total_step, reward, color="red")
	plt.savefig("{}/{}/log/training_reward.png".format(directory_path, comment))
	plt.figure()
	plt.plot(total_step, episode_step, color="blue")
	plt.savefig("{}/{}/log/training_step.png".format(directory_path, comment))

def make_loss_graph(directory_path, comment, fixed_q_update_counter):
	df = pd.read_csv("{}/{}/loss/{}_loss.csv".format(directory_path, comment, fixed_q_update_counter))
	total_step = np.array(df.loc[:, "total_step"].values, dtype=np.int)
	loss_mean = np.array(df.loc[:, "loss_mean"].values, dtype=np.float)
	loss_std = np.array(df.loc[:, "loss_std"].values, dtype=np.float)
	penalty_mean = np.array(df.loc[:, "penalty_mean"].values, dtype=np.float)
	penalty_std = np.array(df.loc[:, "penalty_std"].values, dtype=np.float)
	plt.figure()
	plt.plot(total_step, loss_mean, color="red")
	plt.fill_between(total_step, loss_mean+loss_std, loss_mean-loss_std, facecolor='red', alpha=0.3)
	plt.savefig("{}/{}/loss/{}_loss.png".format(directory_path, comment, fixed_q_update_counter))
	plt.figure()
	plt.plot(total_step, penalty_mean, color="blue")
	plt.fill_between(total_step, penalty_mean+penalty_std, penalty_mean-penalty_std, facecolor='blue', alpha=0.3)
	plt.savefig("{}/{}/loss/{}_penalty.png".format(directory_path, comment, fixed_q_update_counter))

def separate_data(pic_kind, training_size, test_size):
	if pic_kind == "mnist":
		mnist = fetch_mldata('MNIST original', data_home=".")
		mnist.data   = mnist.data.astype(np.float32)
		mnist.target = mnist.target.astype(np.int32)
		indices = list(np.random.permutation(70000))
		finish_flag = [0 for i in range(10)]
		pics = [[] for i in range(10)]
		for i in indices:
			number = mnist.target[i]
			if len(pics[number]) < (training_size + test_size):
				pics[number].append(mnist.data[i])
				if len(pics[number]) == (training_size + test_size):
					finish_flag[number] += 1
			if all(finish_flag):
				break
		pics = np.array(pics)
	return np.array(pics[:,0:training_size]), np.array(pics[:,training_size:(training_size+test_size)])

def print_result(episode, steps, episode_reward, episode_time, epsilon, total_step, run_time):
	print("-------------------Episode {} finished after {} steps-------------------".format(episode+1, steps+1))
	print ("episode_reward : {}".format(episode_reward))
	print ("episode_time: {}".format(episode_time))
	print ("epsilon : {}".format(epsilon))
	print ("total_step : {}".format(total_step))
	print ("run_time: {}".format(run_time))

if __name__ == "__main__":
	run(args)
