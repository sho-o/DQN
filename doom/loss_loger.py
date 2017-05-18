class Loss_Log():
	def __init__(self, comment, iteration):
		self.comment = comment
		self.iteration = iteration
		f = open("result/{}/loss/loss.csv".format(comment), "a")
		f.write("episode,steps,total_step,loss,penalty\n")
		f.close()

	def __call__(self, episode, steps, total_step, agt):
		sum_loss = 0
		sum_penalty = 0
		for i in range(self.iteration):
			s, a, r, new_s, done = agt.make_minibatch(total_step)
			loss, penalty = agt.compute_loss(s, a, r, new_s, done, loss_log=True)
			sum_loss += loss
			sum_penalty += penalty
		ave_loss = sum_loss/self.iteration
		ave_penalty = sum_penalty/self.iteration
		f = open("result/{}/loss/loss.csv".format(self.comment), "a")
		f.write(str(episode+1) + "," + str(steps) + "," + str(total_step) + "," + str(ave_loss) + "," + str(ave_penalty) + "\n")
		f.close()