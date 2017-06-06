import numpy as np
from chainer import cuda

class Loss_Log():
	def __init__(self, comment, iteration, gpu):
		self.comment = comment
		self.iteration = iteration
		self.gpu = gpu

	def __call__(self, fixed_q_update_counter, total_step, agt):
		loss_list = []
		penalty_list = []
		for i in range(self.iteration):
			s, a, r, new_s, done = agt.make_minibatch(total_step)
			loss, penalty = agt.compute_loss(s, a, r, new_s, done, loss_log=True)
			if self.gpu >= 0:
				loss = cuda.to_cpu(loss)
				penalty = cuda.to_cpu(penalty)
			loss_list.append(loss)
			penalty_list.append(penalty)

		loss_array = np.array(loss_list)
		penalty_array = np.array(penalty_list)
		loss_mean = np.average(loss_array)
		loss_std = np.std(loss_array)
		penalty_mean = np.average(penalty_array)
		penalty_std = np.std(penalty_array)
		self.make_loss_log(self.comment, fixed_q_update_counter, total_step, loss_mean, loss_std, penalty_mean, penalty_std)

	def make_loss_log(self, comment, fixed_q_update_counter, total_step, loss_mean, loss_std, penalty_mean, penalty_std):
		f = open("result/{}/loss/{}_loss.csv".format(self.comment, fixed_q_update_counter), "a")
		f.write(str(fixed_q_update_counter) + "," + str(total_step) + "," + str(loss_mean) + "," + str(loss_std) + "," + str(penalty_mean) + "," + str(penalty_std) + "\n")
		f.close()