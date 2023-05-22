import torch
import torch.utils.data as data
import torch.nn.functional as F 
from Utils.utils import *
import numpy as np

def is_visited(base_dict, user_id, item_id):
	if user_id in base_dict and item_id in base_dict[user_id]:
		return True
	else:
		return False


class Benchmark_test_dataset(data.Dataset):
	def __init__(self, user_num, item_num, train_mat, valid_mat, test_mat, gpu, batch_size=1024):
		super(Benchmark_test_dataset, self).__init__()

		self.user_num = user_num
		self.item_num = item_num
		self.user_list = torch.LongTensor([i for i in range(user_num)])
		self.gpu = gpu

		self.valid_mat = valid_mat
		self.test_mat = test_mat
		self.batch_size = batch_size

		self.generate_IDCG_values()
		self.generate_eval_batches(train_mat)
		
		self.max_num = {'valid': max([len(valid_mat[x]) for x in valid_mat]), 'test': max([len(test_mat[x]) for x in test_mat])}

	def generate_IDCG_values(self):

		max_item_num = max(max([len(self.valid_mat[user]) for user in self.valid_mat]), max([len(self.test_mat[user]) for user in self.test_mat]))

		valid_IDCG = []
		test_IDCG = []

		self.denominator = 1 / torch.log2(torch.arange(max_item_num) + 2)

		for user in range(self.user_num):
			if user not in self.valid_mat: 
				valid_IDCG.append(-1)
				continue
			numerator = (2 ** torch.FloatTensor(sorted(self.valid_mat[user].values(), reverse=True)) - 1)
			idcg_user = float((numerator * self.denominator[:numerator.size()[0]]).sum())
			valid_IDCG.append(idcg_user)

		for user in range(self.user_num):
			if user not in self.test_mat: 
				test_IDCG.append(-1)
				continue
			numerator = (2 ** torch.FloatTensor(sorted(self.test_mat[user].values(), reverse=True)) - 1)
			idcg_user = float((numerator * self.denominator[:numerator.size()[0]]).sum())
			test_IDCG.append(idcg_user)

		self.IDCG = {'valid': valid_IDCG, 'test': test_IDCG}


	def generate_eval_batches(self, train_mat):
		
		self.batches = {'user':{}, 'user_degree': {}, 'train':{}, 'valid':{}, 'test':{}, 'IDCG': {'valid':{}, 'test':{}}}

		if self.user_num % self.batch_size == 0:
			num_batches = self.user_num // self.batch_size
		else:
			num_batches = (self.user_num // self.batch_size) + 1

		for batch_id in range(num_batches):
			
			start_idx = batch_id * self.batch_size
			end_idx = min((batch_id + 1) * self.batch_size, self.user_num)

			batch_user_list = self.user_list[start_idx: end_idx]

			valid_y_list, test_y_list = [], []
			train_y_list, train_user_degree_list = [], []
			valid_IDCG_list, test_IDCG_list = [], []

			for batch_user_idx in range(batch_user_list.size(0)):
				real_user_id = int(batch_user_list[batch_user_idx])
				valid_IDCG_list.append(self.IDCG['valid'][real_user_id])
				test_IDCG_list.append(self.IDCG['test'][real_user_id])

				if real_user_id in self.valid_mat:
					for item_id in self.valid_mat[real_user_id]:
						valid_y_list.append((batch_user_idx, item_id, self.valid_mat[real_user_id][item_id]))

				if real_user_id in self.test_mat:
					for item_id in self.test_mat[real_user_id]:
						test_y_list.append((batch_user_idx, item_id, self.test_mat[real_user_id][item_id]))

				if real_user_id in train_mat:
					for item_id in train_mat[real_user_id]:
						train_y_list.append((batch_user_idx, item_id, train_mat[real_user_id][item_id]))

					train_user_degree_list.append(len(train_mat[real_user_id]))
				else:
					train_user_degree_list.append(0)


			self.batches['user'][batch_id] = torch.LongTensor(batch_user_list).to(self.gpu)
			self.batches['user_degree'][batch_id] = torch.LongTensor(train_user_degree_list)

			self.batches['valid'][batch_id] = torch.LongTensor(valid_y_list)
			self.batches['test'][batch_id] = torch.LongTensor(test_y_list)
			self.batches['train'][batch_id] = torch.LongTensor(train_y_list)
			
			self.batches['IDCG']['valid'][batch_id] = torch.FloatTensor(valid_IDCG_list).to(self.gpu)
			self.batches['IDCG']['test'][batch_id] = torch.FloatTensor(test_IDCG_list).to(self.gpu)


class BDL_dataset(data.Dataset):
	def __init__(self, user_num, item_num, rating_mat, interactions, B_mat, S_01_mat, S_01):
		super(BDL_dataset, self).__init__()
		
		self.user_num = user_num
		self.item_num = item_num
		self.rating_mat = rating_mat
		self.interactions = interactions
		self.num_unrated = 1

		self.set_B_mat(B_mat, S_01_mat)
		self.S_01_mat = S_01_mat
		self.S_01 = S_01
		self.train_arr = None
		

	def dataset_construction(self):
		
		self.train_arr = []
		sample_list = np.random.choice(list(range(self.item_num)), size = 10 * (len(self.interactions) + len(self.S_01))* self.num_unrated)
		
		sample_idx = 0
		for user, pos_item, rating in self.interactions:
			ns_count = 0
			
			while True:
				neg_item = sample_list[sample_idx]
				if not is_visited(self.rating_mat, user, neg_item) and not is_visited(self.S_01_mat, user, neg_item):
					self.train_arr.append((user, pos_item, neg_item, rating))
					sample_idx += 1
					ns_count += 1
					if ns_count == self.num_unrated:
						break
						
				sample_idx += 1

		for user, pos_item, rating in self.S_01:
			ns_count = 0
			
			while True:
				neg_item = sample_list[sample_idx]
				if not is_visited(self.rating_mat, user, neg_item) and not is_visited(self.S_01_mat, user, neg_item):
					self.train_arr.append((user, pos_item, neg_item, rating))
					sample_idx += 1
					ns_count += 1
					if ns_count == self.num_unrated:
						break
						
				sample_idx += 1

		self.train_arr = torch.FloatTensor(self.train_arr)

	
	def __len__(self):
		if self.train_arr == None:
			return len(self.interactions) * self.num_unrated
		else:
			return len(self.train_arr)
		
	def __getitem__(self, idx):

		return {'user': self.train_arr[idx][0], 
				'item_b': self.train_arr[idx][1], 
				'item_ub': self.train_arr[idx][2],
				'rating': self.train_arr[idx][3]}


	def set_B_mat(self, B_mat, S_01_mat):

		self.B_mat = B_mat
		for u in self.rating_mat:
			for i in self.rating_mat[u]:
				self.B_mat[u][i] = 1.
		for u in S_01_mat:
			for i in S_01_mat[u]:
				self.B_mat[u][i] = 1.


	def get_affinity_group(self, user_idx, item_idx):
		return self.B_mat[user_idx, item_idx]

