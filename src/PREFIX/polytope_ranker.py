import numpy as np
from numpy.random import default_rng
import pandas as pd
import argparse

import copy 
import math

from itertools import product
from collections import defaultdict, Counter

from PREFIX.prefix_lattice_point_sampler import PrefixLatticePointSampler



class PrefixLatticeSampler(object):
	"""docstring for ClassName"""

	def __init__(self, true_ranking, true_scores, id_2_group, num_groups, proportions, p_deviation, nsteps, flag, k):
		"""initialize the class and its corresponding DP table"""

		self.true_ranking = true_ranking
		self.true_scores = true_scores
		self.id_2_group = id_2_group

		self.rank_len = k
		self.blocksize = int(k/nsteps)
		self.num_groups = num_groups
		self.nsteps = nsteps
		self.proportions = proportions
		self.flag = flag
		self.intra_group_ranking = self.get_intra_group_ranking()
		

		self.LB, self.UB = self.get_bounds(p_deviation)
		assert len(self.LB) == nsteps

		self.sampler = PrefixLatticePointSampler(self.num_groups, self.rank_len, self.LB, self.UB, self.nsteps, self.blocksize)




	def get_intra_group_ranking(self):

		intra_group_ranking = defaultdict(list)

		counter = np.zeros(self.num_groups)
		for item in self.true_ranking:
			intra_group_ranking[self.id_2_group[item]].append(item)
			counter[self.id_2_group[item]] += 1
			if (counter > self.rank_len).all():
				break

		return intra_group_ranking


	def get_bounds(self, delta):

		# Generating L_k and U_k vectors
		L = []
		U = []
		for i in range(self.nsteps):
			if i < self.nsteps - 1:
				prefix_delta = delta/(self.nsteps - i)
			else:
				prefix_delta = delta
			print("delta: ", prefix_delta)
			L_i = []
			U_i = []
			for j in range(self.num_groups):
				L_i.append(max(0,math.ceil((self.proportions[j]-prefix_delta)*(i+1)*self.blocksize)))
				U_i.append(math.floor((self.proportions[j]+prefix_delta)*(i+1)*self.blocksize))
			L.append(L_i)
			U.append(U_i)
		print(L) 
		print(U)
		return L, U


	def construct_ranking(self, answer):

		final_ranking = []

		intra_group_ranking = copy.deepcopy(self.intra_group_ranking)

		for i, item in enumerate(answer):
			final_ranking.append(intra_group_ranking[item].pop(0))

		return final_ranking


	def sample_ranking(self, num_samples):
		all_sampled_points = np.array(self.sampler.sample(num_samples), dtype=int)
		
		assert len(all_sampled_points) == num_samples


		final_rankings = []
		for idx, sampled_point in enumerate(all_sampled_points):
			
			# Get a random permutation of groups for the ranking
			for rank in range(len(sampled_point)):
				permutation = []
				for group in range(self.num_groups):
					permutation += [group]*sampled_point[rank][group]

				np.random.shuffle(permutation)


				# Need to map the group-wise ranking to  document ids
				if rank == 0:
					prefix_permutation = permutation
				else:
					prefix_permutation += permutation
			final_rankings.append(self.construct_ranking(prefix_permutation))
		return final_rankings
		


