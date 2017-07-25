import numpy as np
import agent
import preprocess
from chainer import cuda, serializers
import argparse
import time
import os
import sys
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pylab as plt
import gym
import multiprocessing
import loss_loger
import evaluation
import pandas as pd
import random

parser = argparse.ArgumentParser()
parser.add_argument('--game', '-G', default='atari', type=str, help='game type (doom or atari)')
parser.add_argument('--name', '-N', default='defaut', type=str, help='game name')
parser.add_argument('--directory_path', '-dp', default="result", type=str, help='directory path')
parser.add_argument('--render', '-rd', type=bool, default=False, help='rendor or not')
parser.add_argument('--comment', '-c', default='', type=str, help='comment to distinguish output')
parser.add_argument('--gpu', '-g', default= -1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--exp_policy', '-p', default="epsilon_greedy", type=str, help='explorlation policy')
parser.add_argument('--epsilon_decrease_end', '-ee', default=10**6, type=int, help='the step number of the end of epsilon decrease')
parser.add_argument('--max_episode', '-e', default=10**7, type=int, help='number of episode to learn')
parser.add_argument('--max_step', '-s', default=10000, type=int, help='max steps per episode')
parser.add_argument('--finish_step', '-fs', default=2*10**7, type=int, help='end of the learning')
parser.add_argument('--q_update_freq', '-q', default=4, type=int, help='q update freaquency')
parser.add_argument('--fixed_q_update_freq', '-f', default=10**4, type=int, help='fixed q update freaquency')
parser.add_argument('--save_freq', '-sf', default=10**5, type=int, help='save frequency')
parser.add_argument('--eval_freq', '-ef', default=10**5, type=int, help='evaluatuin frequency')
parser.add_argument('--print_freq', '-pf', default=1, type=int, help='print result frequency')
parser.add_argument('--graph_freq', '-gf', default=2*10**6, type=int, help='make graph frequency')
parser.add_argument('--memory_save_freq', '-mf', default=5*10**6, type=int, help='memory save frequency')
parser.add_argument('--initial_exploration', '-i', default=5*10**4, type=int, help='number of initial exploration')
parser.add_argument('--batch_size', '-b', type=int, default=32, help='learning minibatch size')
parser.add_argument('--memory_size', '-ms', type=int, default=10**6, help='replay memory size')
parser.add_argument('--input_slides', '-is', type=int, default=4, help='number of input slides')
parser.add_argument('--net_type', '-n', type=str, default="DQN", help='network type')
parser.add_argument('--pic_size', '-ps', type=int, default=84, help='nput pic size')
parser.add_argument('--discount', '-d', type=float, default=0.99, help='discount factor')
parser.add_argument('--rms_eps', '-re', type=float, default=0.01, help='RMSProp_epsilon')
parser.add_argument('--rms_lr', '-lr', type=float, default=0.00025, help='RMSProp_learning_rate')
parser.add_argument('--optimizer_type', '-o', type=str, default="rmsprop", help='type of optimizer')
parser.add_argument('--mode', '-m', type=str, default="default", help='default or regularize or mix')
parser.add_argument('--threshold', '-t', type=float , default=0.001, help='regularization threshold')
parser.add_argument('--penalty_weight', '-pw', type=float, default=1.0, help='regularization penalty weight')
parser.add_argument('--mix_rate', '-mr', type=float, default=0, help='target_mix _rate')
parser.add_argument('--loss_log_iter', '-li', type=int, default=10, help='(batch) iteration  compute average loss and penalty (1batch=32)')
parser.add_argument('--loss_log_freq', '-lf', default=1000000, type=int, help='record loss frequency per step')
parser.add_argument('--loss_log_length', '-ll', default=1000, type=int, help='record loss step length')parser.add_argument('--rolling_mean_width', '-r', default=1000, type=int, help='width of rolling mean')
parser.add_argument('--skip_size', '-ss', type=int, default=4, help='skip size')
parser.add_argument('--num_of_actions', '-na', type=int, default=4, help='number of actions')
parser.add_argument('--eval_iter', '-ei', type=int, default=30, help='iteration of evaluation')
parser.add_argument('--max_initial_noop', '-mn', type=int, default=30, help='maximum times of initial noop')
parser.add_argument('--penalty_function', '-pvf', type=str, default="action_value", choices=['value', 'action_value', 'max_action_value'], help='value function type used to compute penatlty')
parser.add_argument('--penalty_type', '-pt', type=str, default="huber", choices=['huber', 'mean_squared'], help='penalty error function type')
parser.add_argument('--seed', '-sd', type=int, default=0, help='random seed')

args = parser.parse_args()

def run(args):
	game = args.game
	if game == "doom":
		import ppaquette_gym_doom
	name = args.name
	if name == "defaut":
		if game == "doom":
			name = 'ppaquette/DoomDefendCenter-v0'
		if game == "atari":
			name = 'PongDeterministic-v0'
	render = args.render
	comment = args.comment
	gpu = args.gpu
	directory_path = args.directory_path
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
	memory_save_freq = args.memory_save_freq
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
	mode = args.mode
	threshold = args.threshold
	mix_rate = args.mix_rate
	loss_log_iter = args.loss_log_iter
	penalty_weight = args.penalty_weight
	loss_log_freq = args.loss_log_freq
	loss_log_length = args.loss_log_length
	rolling_mean_width = args.rolling_mean_width
	skip_size = args.skip_size
	num_of_actions = args.num_of_actions
	eval_iter = args.eval_iter
	max_initial_noop = args.max_initial_noop
	penalty_function = args.penalty_function
	penalty_type = args.penalty_type
	seed = args.seed
	epsilon_decrease_wide = 0.9/(epsilon_decrease_end - initial_exploration)

	random.seed(seed)
	np.random.seed(seed)
	if gpu >= 0:
		cuda.get_device(gpu).use()
		cuda.cupy.random.seed(seed)

	make_directries(directory_path, comment, ["network", "log", "evaluation", "loss", "replay_memory"])
	pre = preprocess.Preprocess()
	agt = agent.Agent(exp_policy, net_type, gpu, pic_size, num_of_actions, memory_size, input_slides, batch_size, discount, rms_eps, rms_lr, optimizer_type, mode, threshold, penalty_weight, mix_rate, penalty_function, penalty_type)
	env = gym.make(name)
	#multiprocessing_lock = multiprocessing.Lock()
	#env.configure(lock=multiprocessing_lock)
	eva = evaluation.Evaluation(directory_path, game, name, comment, max_step, skip_size, eval_iter, max_initial_noop)
	loss_log = loss_loger.Loss_Log(directory_path, comment, loss_log_iter, gpu)
	total_step = 0
	fixed_q_update_counter = 0
	loss_log_flag = 0
	if game == "atari":
		num_of_actions = env.action_space.n
	run_start = time.time()

	for episode in range(max_episode):
		episode_start = time.time()
		episode_reward = 0
		episode_value = 0
		obs = env.reset()
		s = np.zeros((4, 84, 84), dtype=np.uint8)
		s[3] = pre.one(obs)

		if total_step > finish_step:
			memory_save(game, directory_path, comment, total_step, agt, gpu)
			break

		for steps in range(max_step):
			if render == True:
				env.render()

			a, value = agt.policy(s)
			if game == "doom":
				action = pre.action_convert(a)
				obs, r, done, info = env.step(action)
			if game == "atari":
				obs, r, done, info = env.step(a)
			obs_processed = pre.one(obs)

			new_s = np.asanyarray([s[1], s[2], s[3], obs_processed], dtype=np.uint8)
			r_clipped = pre.reward_clip(r)
			agt.store_experience(total_step, s, a, r_clipped, new_s, done)

			episode_reward += r
			episode_value += value
			total_step += 1

			#update and save
			if total_step > initial_exploration:
				if total_step % q_update_freq == 0:
					agt.q_update(total_step)
				if (total_step+1) % loss_log_freq == 0:
					make_loss_log_file(directory_path, comment, total_step+1)
					loss_log_counter = 0
					loss_log_step = total_step+1
					loss_log_flag = 1
				if loss_log_flag == 1:
					loss_log(loss_log_step, fixed_q_update_counter, total_step, agt)
					loss_log_counter += 1
					if loss_log_counter % loss_log_length == 0:
						loss_log_flag = 0
				if total_step % fixed_q_update_freq == 0:
					print "----------------------- fixed Q update ------------------------------"
					agt.fixed_q_updqte()
					fixed_q_update_counter += 1
					#if fixed_q_update_counter % loss_log_freq == 0:
					#	make_loss_log_file(directory_path, comment, fixed_q_update_counter)
					#if fixed_q_update_counter % loss_log_freq == 1 and fixed_q_update_counter > 1:
					#	print "----------------------- make_loss_graph ------------------------------"
					#	make_loss_graph(directory_path, comment, fixed_q_update_counter-1)
				if total_step % save_freq == 0:
					print "----------------------- save the_model ------------------------------"
					serializers.save_npz('{}/{}/network/q_{}.net'.format(directory_path, comment, total_step), agt.q)
				if total_step % eval_freq == 0:
					print "----------------------- evaluate the model ------------------------------"
					eva(agt, pre, episode, total_step)
				if total_step % memory_save_freq == 0:
					print "----------------------- saving replay memory ------------------------------"
					memory_save(game, directory_path, comment, total_step, agt)
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
			s = np.array(new_s)

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

def memory_save(game, directory_path, comment, total_step, agt):
	mem_kinds = ["s", "a", "r", "new_s", "done"]
	for k in mem_kinds:
		if os.path.exists("/disk/waikiki/ohnishi-s/"):
			if not os.path.exists("/disk/waikiki/ohnishi-s/{}/{}".format(game, directory_path)):
				os.makedirs("/disk/waikiki/ohnishi-s/{}/{}".format(game, directory_path))
			path = '/disk/waikiki/ohnishi-s/{}/{}/{}_{}'.format(game, directory_path, comment, k)
		else:
			path = '{}/{}/replay_memory/{}'.format(directory_path, comment, k)
		np.save(path, agt.replay_memory[k][:total_step])

def make_test_graph(directory_path, comment):
	df = pd.read_csv("{}/{}/evaluation/evaluation.csv".format(directory_path, comment))
	total_step = np.array(df.loc[:, "total_step"].values, dtype=np.int)
	reward_mean = np.array(df.loc[:, "reward_mean"].values, dtype=np.float)
	reward_std = np.array(df.loc[:, "reward_std"].values, dtype=np.float)
	#step_mean = np.array(df.loc[:, "step_mean"].values, dtype=np.float)
	#step_std = np.array(df.loc[:, "step_std"].values, dtype=np.float)
	plt.figure()
	plt.plot(total_step, reward_mean, color="red")
	plt.fill_between(total_step, reward_mean+reward_std, reward_mean-reward_std, facecolor='red', alpha=0.3)
	plt.savefig("{}/{}/evaluation/reward.png".format(directory_path, comment))
	#plt.figure()
	#plt.plot(total_step, step_mean, color="blue")
	#plt.fill_between(total_step, step_mean+step_std, step_mean-step_std, facecolor='blue', alpha=0.3)
	#plt.savefig("result/{}/evaluation/step.png".format(comment))

def make_training_graph(directory_path, comment, rolling_mean_width):
	df = pd.read_csv("{}/{}/log/log.csv".format(directory_path, comment))
	total_step = np.array(df.loc[:, "total_step"].values, dtype=np.int)
	reward = np.array(df.loc[:, "reward"].values, dtype=np.int)
	reward = pd.Series(reward).rolling(window=rolling_mean_width).mean()
	#episode_step = np.array(df.loc[:, "episode_step"].values, dtype=np.float)
	#episode_step = pd.Series(episode_step).rolling(window=rolling_mean_width).mean()
	plt.figure()
	plt.plot(total_step, reward, color="red")
	plt.savefig("{}/{}/log/training_reward.png".format(directory_path, comment))
	#plt.figure()
	#plt.plot(total_step, episode_step, color="blue")
	#plt.savefig("result/{}/log/training_step.png".format(comment))

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

def print_result(episode, steps, episode_reward, episode_time, epsilon, total_step, run_time):
	print("-------------------Episode {} finished after {} steps-------------------".format(episode+1, steps+1))
	print ("episode_reward : {}".format(episode_reward))
	print ("episode_time: {}".format(episode_time))
	print ("epsilon : {}".format(epsilon))
	print ("total_step : {}".format(total_step))
	print ("run_time: {}".format(run_time))
	print ("time/step: {}".format(episode_time/steps))

if __name__ == "__main__":
	run(args)
