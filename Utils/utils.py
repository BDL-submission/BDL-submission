import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from Utils.evaluation import *
import numpy as np


#%% Dataset load
def read_data(path, data_name='yahoo'):

	np.random.seed(0)
	valid_ratio = 0.1

	user_count, item_count = -1, -1

	train_tmp, test_tmp = [], []
	with open(path + data_name + '_user.txt', 'r') as f:
		for line in f.readlines():
			u, i, r = line.strip().split(',')
			user_count = max(int(u), user_count)
			item_count = max(int(i), item_count)
			train_tmp.append([int(u), int(i), int(r)])

	with open(path + data_name + '_random.txt', 'r') as f:
		for line in f.readlines():
			u, i, r = line.strip().split(',')
			user_count = max(int(u), user_count)
			item_count = max(int(i), item_count)
			test_tmp.append([int(u), int(i), int(r)])


	total_mat = list_to_dict(train_tmp)
	test_mat = list_to_dict(test_tmp)
	train_mat, valid_mat = {}, {}

	for user in total_mat:
		items = list(total_mat[user].keys())
		np.random.shuffle(items)

		num_test_items = int(len(items) * valid_ratio)
		valid_items = items[:num_test_items]
		train_items = items[num_test_items:]

		for item in valid_items:
			dict_set(valid_mat, user, item, total_mat[user][item])

		for item in train_items:
			dict_set(train_mat, user, item, total_mat[user][item])

	train_mat_R = {}

	for user in train_mat:
		for item in train_mat[user]:
			dict_set(train_mat_R, item, user, train_mat[user][item])

	train_interactions = []
	for user in train_mat:
		for item in train_mat[user]:
			train_interactions.append([user, item, train_mat[user][item]])
			
	return user_count + 1, item_count + 1, train_mat, train_interactions, valid_mat, test_mat


def xavier_normal_initialization(module):
    r"""using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
	    

#%% helper function
def to_np(x):
	return x.data.cpu().numpy()


def dict_set(base_dict, user_id, item_id, val):
	if user_id in base_dict:
		base_dict[user_id][item_id] = val
	else:
		base_dict[user_id] = {item_id: val}


def is_visited(base_dict, user_id, item_id):
	if user_id in base_dict and item_id in base_dict[user_id]:
		return True
	else:
		return False


def list_to_dict(base_list):
	result = {}
	for user_id, item_id, value in base_list:
		dict_set(result, user_id, item_id, value)
	
	return result


def dict_to_list(base_dict):
	result = []

	for user_id in base_dict:
		for item_id in base_dict[user_id]:
			result.append((user_id, item_id, base_dict[user_id][item_id]))
	
	return result