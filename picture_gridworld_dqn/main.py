import numpy as np
import agent
import environment
from chainer import cuda, serializers
import argparse
import time
import os
import sys
import evaluation
import loss_penalty_log

parser = argparse.ArgumentParser()
parser.add_argument('--comment', '-c', default='', type=str, help='comment to distinguish output')
parser.add_argument('--gpu', '-g', default= -1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--pic_kind', '-k', default="mnist", type=str, help='kind of pictures')
parser.add_argument('--exp_policy', '-p', default="epsilon_greedy", type=str, help='explorlation policy')
parser.add_argument('--epsilon_decrease_end', '-ee', default=10**6, type=int, help='the step number of the end of epsilon decrease')
parser.add_argument('--max_episode', '-e', default=10**7, type=int, help='number of episode to learn')
parser.add_argument('--max_step', '-s', default=1000, type=int, help='max steps per episode')
parser.add_argument('--finish_step', '-fs', default=5*10**7, type=int, help='end of the learning')
parser.add_argument('--q_update_freq', '-q', default=4, type=int, help='q update freaquency')
parser.add_argument('--fixed_q_update_freq', '-f', default=10**4, type=int, help='fixed q update frequency')
parser.add_argument('--save_freq', '-sf', default=5*10**4, type=int, help='save frequency')
parser.add_argument('--eval_freq', '-ef', default=10**4, type=int, help='evaluatuin frequency')
parser.add_argument('--print_freq', '-pf', default=1, type=int, help='print result frequency')
parser.add_argument('--initial_exploration', '-i', default=5*10**4, type=int, help='number of initial exploration')
parser.add_argument('--batch_size', '-b', type=int, default=32, help='learning minibatch size')
parser.add_argument('--memory_size', '-m', type=int, default=10**6, help='replay memory size')
parser.add_argument('--input_slides', '-is', type=int, default=1, help='number of input slides')
parser.add_argument('--net_type', '-n', type=str, default="full_connect", help='network type')
parser.add_argument('--pic_size', '-ps', type=int, default=28, help='nput pic size')
parser.add_argument('--discount', '-d', type=float, default=0.99, help='discount factor')
parser.add_argument('--rms_eps', '-re', type=float, default=0.01, help='RMSProp_epsilon')
parser.add_argument('--rms_lr', '-lr', type=float, default=0.00025, help='RMSProp_learning_rate')
parser.add_argument('--optimizer_type', '-o', type=str, default="rmsprop", help='type of optimizer')
parser.add_argument('--start_point', '-sp', type=int, default=1, help='start point')
parser.add_argument('--regularize', '-r', type=bool, default=False, help='regularize or not')
parser.add_argument('--threshold', '-t', type=float , default=0.001, help='regularization threshold')
parser.add_argument('--penalty_weight', '-pw', type=float, default=1.0, help='regularization penalty weight')
parser.add_argument('--rlp_iter', '-di', type=int, default=10, help='(batch) iteration for compute average loss and penalty (1batch=32)')
parser.add_argument('--training_pics', '-tp', type=int, default=20, help='number of kinds of training pictures')
args = parser.parse_args()

def run(args):
	comment = args.comment
	gpu = args.gpu
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
	regularize = args.regularize
	threshold = args.threshold
	rlp_iter = args.rlp_iter
	penalty_weight = args.penalty_weight
	training_pics = args.training_pics
	s_init = [(start_point-1)%3, (start_point-1)/3]
	epsilon_decrease_wide = 0.9/(epsilon_decrease_end - initial_exploration)

	run_start = time.time()
	make_directries(comment, ["network", "log", "evaluation", "loss_and_penalty"])
	if gpu >= 0:
		cuda.get_device(gpu).use()
	env = environment.Environment(pic_kind, training_pics)
	actions = ["up", "down", "right", "left"] 
	num_of_actions = len(actions)
	agt = agent.Agent(exp_policy, net_type, gpu, pic_size, num_of_actions, memory_size, input_slides, batch_size, discount, rms_eps, rms_lr, optimizer_type, regularize, threshold, penalty_weight)
	eva = evaluation.Evaluation(comment, pic_kind, s_init, actions, max_step)
	rlp = loss_penalty_log.RLP_Log(comment, rlp_iter)
	total_step = 0
	
	for episode in range(max_episode):
		episode_start = time.time()
		env.make_episode_pics()
		episode_reward = 0
		episode_value = 0
		s = s_init
		pic_s = env.s_to_pic(s)
		if total_step >= finish_step:
			break

		for steps in range(max_step):
			a, value = agt.policy(pic_s)
			new_s = env.generate_next_s(s, actions[a])
			pic_new_s = env.s_to_pic(new_s)
			r = env.make_reward(s, actions[a])
			done = env.judge_finish(new_s)
			agt.store_experience(total_step, pic_s, a, r, pic_new_s, done)
			episode_reward += r
			episode_value += value
			total_step += 1

			#update and save
			if total_step > initial_exploration:
				if total_step % q_update_freq == 0:
					agt.q_update(total_step)
				if total_step % fixed_q_update_freq == 0:
					print "----------------------- record loss and penalty ------------------------------"
					rlp(episode, total_step, agt)
					print "----------------------- fixed Q update ------------------------------"
					agt.fixed_q_updqte()
				if total_step % save_freq == 0:
					print "----------------------- save the_model ------------------------------"
					serializers.save_npz('result/{}/network/q.net'.format(comment), agt.q)
				if total_step % eval_freq == 0:
					print "----------------------- evaluate the_model ------------------------------"
					eva(agt, episode, total_step)
				agt.epsilon = max(0.1, agt.epsilon - epsilon_decrease_wide)

			#log, print_result
			if done:
				run_time = time.time() - run_start
				episode_time = time.time() - episode_start
				episode_average_value = episode_value/steps
				make_log(comment, episode, episode_reward, episode_average_value, agt.epsilon, steps, total_step, run_time)
				if total_step % print_freq == 0:
					print_result(episode, steps, episode_reward, episode_time, agt.epsilon, total_step, run_time)
				break

			#prepare next s
			s = new_s[:]
			pic_s = np.array(pic_new_s)

def make_directries(comment, dirs):
	for d in dirs:
		if not os.path.exists("result/{}/".format(comment) + d):
			os.makedirs("result/{}/".format(comment) + d)

def make_log(comment, episode, episode_reward, episode_average_value, epsilon, steps, total_step, run_time):
	f = open("result/{}/log/log.csv".format(comment), "a")
	if episode == 0:
		f.write("episode,reward,average_value,epsilon,episode_step,total_step,run_time\n")
	f.write(str(episode+1) + "," + str(episode_reward) + "," + str(episode_average_value) + "," + str(epsilon) + ',' + str(steps+1) + ',' + str(total_step) + ',' + str(run_time) + "\n")
	f.close()

def print_result(episode, steps, episode_reward, episode_time, epsilon, total_step, run_time):
	print("-------------------Episode {} finished after {} steps-------------------".format(episode+1, steps+1))
	print ("episode_reward : {}".format(episode_reward))
	print ("episode_time: {}".format(episode_time))
	print ("epsilon : {}".format(epsilon))
	print ("total_step : {}".format(total_step))
	print ("run_time: {}".format(run_time))

if __name__ == "__main__":
	run(args)
