import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.evaluation import *
from Utils.utils import *

from math import ceil

class backbone(nn.Module):

	def __init__(self, user_num, item_num, dim, gpu):
		super(backbone, self).__init__()
		self.user_num = user_num
		self.item_num = item_num

		self.user_list = torch.LongTensor([i for i in range(user_num)]).to(gpu)
		self.item_list = torch.LongTensor([i for i in range(item_num)]).to(gpu)

		self.user_emb = nn.Embedding(self.user_num, dim)
		self.item_emb = nn.Embedding(self.item_num, dim)
				
		self.classifier = nn.Linear(dim, 5)
		
		self.loss = nn.CrossEntropyLoss()
		self.values = torch.FloatTensor([[1,2,3,4,5]]).to(gpu)
		self.apply(xavier_normal_initialization)
				
		
	def forward_pair(self, user, item):
		h_u = self.user_emb(user)
		h_i = self.item_emb(item)
		h_ui = h_u * h_i
		
		return h_ui


	def get_preference_loss(self, output, gt):
		h_ui = output
		logits = self.classifier(h_ui)
		CE_loss = self.loss(logits, (gt.long()-1).view(-1)) / 5
					 
		return CE_loss
	

	def get_embedding(self):

		user = self.user_emb(self.user_list)
		item = self.item_emb(self.item_list)

		return user, item

		
	def forward_triple(self, user, item, neg_item):
		h_u = self.user_emb(user)
		h_i = self.item_emb(item)
		h_j = self.item_emb(neg_item)

		h_ui = h_u * h_i
		h_uj = h_u * h_j

		return h_ui, h_uj
	

	def forward_full_items(self, batch_user):
		
		bs = batch_user.size(0)
		user = self.user_emb(batch_user)
		item = self.item_emb(self.item_list)
		
		u_d = user.repeat(1, self.item_num).view(bs * self.item_num, -1)
		i_d = item.repeat(bs, 1)
		
		h_ui = u_d * i_d
		
		prob = F.softmax(self.classifier(h_ui), -1)
		return (self.values * prob).sum(-1).view(bs, -1)

	

class BDL(backbone):
	def __init__(self, user_num, item_num, dim, tau, gpu):
		backbone.__init__(self, user_num, item_num, dim, gpu)

		# temporal mean
		self.user_emb_T = nn.Embedding(self.user_num, dim)
		self.item_emb_T = nn.Embedding(self.item_num, dim)
		self.tau = tau
		self._init_target()

	def _init_target(self):        
		for param_o, param_t in zip(self.user_emb.parameters(), self.user_emb_T.parameters()):
			param_t.data.copy_(param_o.data)
			param_t.requires_grad = False
		
		for param_o, param_t in zip(self.item_emb.parameters(), self.item_emb_T.parameters()):
			param_t.data.copy_(param_o.data)
			param_t.requires_grad = False
		
	def _update_target(self):
		for param_o, param_t in zip(self.user_emb.parameters(), self.user_emb_T.parameters()):
			param_t.data = param_t.data * self.tau + param_o.data * (1. - self.tau)
		
		for param_o, param_t in zip(self.item_emb.parameters(), self.item_emb_T.parameters()):
			param_t.data = param_t.data * self.tau + param_o.data * (1. - self.tau)

	def get_KD_loss(self, output, gt):

		h_ui = output
		logits = self.classifier(h_ui)
		
		MSE_loss = (((F.softmax(logits, -1) * self.values).sum(-1) - gt.view(-1)) ** 2)

		return MSE_loss.mean() * 0.2

	def get_conf_with_filter(self, output, r_filter, implement='point-wise'):
		h_ui, h_uj = output[0], output[1]

		y_pos = self.classifier(h_ui).max(-1, keepdim=True)[0]
		y_neg = self.classifier(h_uj).max(-1, keepdim=True)[0]
		
		if implement == 'point-wise':
			return -((y_neg).sigmoid().log() * r_filter).mean() -((-y_pos).sigmoid().log()).mean()
		
		elif implement == 'pair-wise':
			return -((y_neg - y_pos).sigmoid().log() * r_filter).mean()

	
	@torch.no_grad()
	def compute_reliability(self):
		
		reliability_mat = []
		batch_size = 1024
		batch_num = ceil(self.user_list.size(0)/1024)

		for batch_idx in range(batch_num):
			start_id = batch_idx * batch_size
			end_id = min(start_id + batch_size, self.user_list.size(0))
			batch_user = self.user_list[start_id: end_id]
			bs = batch_user.size(0)
			
			# target model
			user = self.user_emb(batch_user)
			item = self.item_emb(self.item_list)

			u_d = user.repeat(1, self.item_num).view(bs * self.item_num, -1)
			i_d = item.repeat(bs, 1)

			h_ui = u_d * i_d
			pred = F.softmax(self.classifier(h_ui), -1).max(-1, keepdim=True)[1].view(bs, -1)
			
			# temporal mean
			user = self.user_emb_T(batch_user)
			item = self.item_emb_T(self.item_list)

			u_d = user.repeat(1, self.item_num).view(bs * self.item_num, -1)
			i_d = item.repeat(bs, 1)

			h_ui = u_d * i_d
			pred_T = F.softmax(self.classifier(h_ui), -1).max(-1, keepdim=True)[1].view(bs, -1)
			
			reliability_mat.append(pred == pred_T)

		return torch.cat(reliability_mat, 0)


class Discriminator(nn.Module):
	def __init__(self, dim):
		super(Discriminator, self).__init__()
		self.clf = nn.Linear(dim, 1)
		self.BCE_loss = nn.BCEWithLogitsLoss()

	def get_adv_loss(self, h, target):

		pred = self.clf(h)
		target = torch.sigmoid(target.detach())
		bce_loss = self.BCE_loss(pred, target.unsqueeze(-1))
		
		return bce_loss
	
	def freeze_clf(self):
		for param in self.clf.parameters():
			param.requires_grad = False
	
	def unfreeze_clf(self):
		for param in self.clf.parameters():
			param.requires_grad = True        