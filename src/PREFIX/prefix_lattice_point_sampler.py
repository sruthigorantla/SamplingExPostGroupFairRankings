import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matlab.engine
from tqdm import tqdm
eng = matlab.engine.start_matlab()
s = eng.genpath('/pathto/PolytopeSamplerMatlab-master')
eng.addpath(s, nargout=0)
import matlab


class PrefixLatticePointSampler:
	def __init__(self,ell,k,lb,ub,nsteps,blocksize):
		self.ell = ell
		self.k = k
		self.lb = lb
		self.ub = ub
		self.nsteps = nsteps
		self.blocksize = blocksize
		

	def sample(self, num_samples):
		done = False
		sample_count = 0
		sampled_points = []
		
		for _ in tqdm(range(num_samples)):
			point = []
			prefix_counts = [0]*self.ell
			for i in range(len(self.lb)):
				lb = []
				ub = []
				for j in range(len(self.lb[i])):
					lb.append(max(0,self.lb[i][j] - prefix_counts[j]))
					ub.append(min(self.blocksize, self.ub[i][j] - prefix_counts[j]))
				self.Delta = self.compute_Delta(lb, ub, i)
				self.x_star = self.get_x_star(lb, ub, i)
				lb_new = ((lb - self.x_star)*(1+(np.sqrt(self.ell)/self.Delta))).reshape(-1,1).tolist()
				ub_new = ((ub - self.x_star)*(1+(np.sqrt(self.ell)/self.Delta))).reshape(-1,1).tolist()
				print(f'block: {i}, Delta: {self.Delta}, lb: {lb}, ub: {ub}')
				while True:
					z = np.transpose(np.asarray(eng.sampling_from_simplex(self.ell,self.blocksize,matlab.double(lb_new),matlab.double(ub_new),2)))[0]
					x = self.round(z)
					if self.inN(x, lb, ub, i):
						point.append(x)
						for j in range(len(x)):
							prefix_counts[j] += x[j]
						break	
			sampled_points.append(point)
		return sampled_points[:num_samples]

	def compute_Delta(self, lb, ub, i):

		
		minimum = np.inf
		for j in range(int(self.ell)):
			minimum = min(minimum, 0.5*(ub[j]-lb[j]) - 1)
		minimum = min(minimum, (self.blocksize - np.sum(lb))/self.ell - 1)
		minimum = min(minimum, (np.sum(ub) - self.blocksize)/self.ell - 1)
		return minimum

	def get_x_star(self, lb, ub, i):
		x = np.zeros(int(self.ell))
		x = lb + np.ceil(self.Delta)
		while np.sum(x) < self.blocksize:
			j = np.argmax(ub - np.ceil(self.Delta) - x)
			x[j] = min( self.blocksize - np.sum(x) + x[j], ub[j] - np.ceil(self.Delta))
		return x

	
	def round(self,z):
		y = z + self.x_star
		sorted_indices = np.argsort(y-np.floor(y))[::-1][:len(z)]

		for j in range(int(self.blocksize - np.sum(np.floor(y)))):
			y[sorted_indices[j]] = np.floor(y[sorted_indices[j]]) + 1
		j = int(self.blocksize - np.sum(np.floor(y)))
		while j < self.ell:
			y[sorted_indices[j]] = np.floor(y[sorted_indices[j]]) 
			j += 1
		return y

	def inN(self,x,lb,ub,i):
		if np.all(np.less_equal(lb,x)) and np.all(np.greater_equal(ub,x)) and np.sum(x) == self.blocksize:
			return True

	def plot(self):
		plt.scatter(s[0], s[2])
		plt.savefig("test.png")


# def main():
# 	o = LatticePointSampler(2,100,[0,2],[5,10])
# 	point = o.sample()
# 	print(point)
# 	exit()

# if __name__ == '__main__':
# 		main()	