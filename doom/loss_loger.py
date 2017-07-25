import numpy as np
from chainer import cuda

class Loss_Log():
	def __init__(self, directory_path, comment, iteration, gpu):
		self.directory_path = directory_path
		self.comment = comment
		self.iteration = iteration
		self.gpu = gpu

	def __call__(self, loss_log_step, fixed_q_update_counter, total_step, agt):
		loss_list = []
		penalty_list = []
		#q_ave_list = []
		#q_std_list = []
		#t_ave_list = []
		#t_std_list = []
		for i in range(self.iteration):
			s, a, r, new_s, done = agt.make_minibatch(total_step)
			print a
			loss, penalty = agt.compute_loss(s, a, r, new_s, done, loss_log=True)
			#loss, penalty, q_qve, q_std, t_ave, t_std = agt.compute_loss(s, a, r, new_s, done, loss_log=True)
			loss_data = loss.data
			penalty_data = penalty.data
			if self.gpu >= 0:
				loss_data = cuda.to_cpu(loss_data)
				penalty_data = cuda.to_cpu(penalty_data)
			loss_list.append(loss_data)
			penalty_list.append(penalty_data)
			#q_ave_list.append(q_ave)
			#q_std_list.append(q_std)
			#t_ave_list.append(t_ave)
			#t_std_list.append(t_std)

		loss_array = np.array(loss_list)
		penalty_array = np.array(penalty_list)
		loss_mean = np.average(loss_array)
		loss_std = np.std(loss_array)
		penalty_mean = np.average(penalty_array)
		penalty_std = np.std(penalty_array)
		#q_ave_ave = np.average(q_ave_list)
		#q_std_ave = np.average(q_std_list)
		#t_ave_ave = np.average(t_ave_list)
		#t_std_ave = np.average(t_std_list)
		self.make_loss_log(loss_log_step, self.directory_path, self.comment, fixed_q_update_counter, total_step, loss_mean, loss_std, penalty_mean, penalty_std)
		#self.make_loss_log(self.directory_path, self.comment, fixed_q_update_counter, total_step, loss_mean, loss_std, penalty_mean, penalty_std, q_ave_ave, q_std_ave, t_ave_ave, t_std_ave)


	def make_loss_log(self, loss_log_step, directory_path, comment, fixed_q_update_counter, total_step, loss_mean, loss_std, penalty_mean, penalty_std):
		f = open("{}/{}/loss/{}_loss.csv".format(directory_path, comment, loss_log_step), "a")
		f.write(str(fixed_q_update_counter) + "," + str(total_step) + "," + str(loss_mean) + "," + str(loss_std) + "," + str(penalty_mean) + "," + str(penalty_std) + "\n")
		#f.write(str(fixed_q_update_counter) + "," + str(total_step) + "," + str(loss_mean) + "," + str(loss_std) + "," + str(penalty_mean) + "," + str(penalty_std) + "," + str(q_ave_ave) + "," + str(q_std_ave) + "," + str(t_ave_ave) + "," + str(t_std_ave) + "\n")
		f.close()