from Utils.utils import *
from Utils.dataset import *

import numpy as np
import torch
import copy


@torch.no_grad()
def densify_interactions(users, items, ratings, batch_size, item_count):
	R = torch.zeros([batch_size, item_count], device=users.device)
	R[users, items] = ratings
	return R


def print_result(epoch, max_epoch, train_loss, eval_results, improved=False):

	if improved:
		print('Epoch [{}/{}], Train Loss: {:.4f}*' .format(epoch, max_epoch, train_loss))
	else: 
		print('Epoch [{}/{}], Train Loss: {:.4f}' .format(epoch, max_epoch, train_loss))

	for mode in ['valid', 'test']:

		RMSE = eval_results[mode]['RMSE']
		MSE = eval_results[mode]['MSE']
		MAE = eval_results[mode]['MAE']
		NDCG = eval_results[mode]['NDCG']

		if mode == 'test':
			mode = 'test '
		print('{}  {}: {:.4f}, {}: {:.4f}, {}: {:.4f}, {}: {:.4f}'.format(mode, 'RMSE', RMSE, 'MSE', MSE, 'MAE', MAE, 'NDCG', NDCG))

	print()


@torch.no_grad()
def evaluate(model, gpu, test_dataset):
	
	metrics = {'RMSE':[], 'MSE':[], 'MAE':[], 'NDCG':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	residual = {'test': [], 'valid': []}
	dcg = {'test': [], 'valid': []}
	ndcg = {'test': [], 'valid': []}

	# score_mat generation
	for batch_id in range(len(test_dataset.batches['user'])):
		batch_user_list = test_dataset.batches['user'][batch_id]
	
		batch_score_mat = model.forward_full_items(batch_user_list)

		for mode in ['test', 'valid']:
			
			if test_dataset.batches[mode][batch_id].size(0) == 0: continue

			batch_user_ids = test_dataset.batches[mode][batch_id][:,0].to(gpu)
			item_ids = test_dataset.batches[mode][batch_id][:,1].to(gpu)
			ratings = test_dataset.batches[mode][batch_id][:,2].type(torch.FloatTensor).to(gpu)
			
			predicted_ratings = batch_score_mat[batch_user_ids, item_ids]
			residual[mode].append(ratings - predicted_ratings)

			# NDCG
			batch_gt_mat_mode = densify_interactions(batch_user_ids, item_ids, ratings, batch_user_list.size(0), model.item_num).type(torch.FloatTensor).to(gpu)
			# batch_score_mat_mask = (batch_score_mat * 1e+5) * (batch_gt_mat_mode > 0)
			batch_score_mat_mask = torch.where(batch_gt_mat_mode > 0, batch_score_mat, torch.tensor(-1000.).to(gpu))
			_, indices_mode = batch_score_mat_mask.topk(test_dataset.max_num[mode], dim=-1)
			batch_prediction_mode = batch_gt_mat_mode.gather(-1, indices_mode)

			numerator = (2 ** batch_prediction_mode - 1)
			dcg_user = (numerator * test_dataset.denominator[:numerator.size()[1]].to(gpu)).sum(-1)
			dcg[mode].append(dcg_user)
			ndcg[mode].append(dcg_user / test_dataset.batches['IDCG'][mode][batch_id])

	# valid, test
	for mode in ['test', 'valid']:

		residual[mode] = torch.cat(residual[mode])
		ndcg[mode] = torch.cat(ndcg[mode])

		eval_results[mode]['RMSE'] = float((residual[mode] ** 2).mean().sqrt())
		eval_results[mode]['MSE'] = float((residual[mode] ** 2).mean())
		eval_results[mode]['MAE'] = float((residual[mode]).abs().mean())
		eval_results[mode]['NDCG'] = float(ndcg[mode][ndcg[mode] > 0].mean())

	return eval_results