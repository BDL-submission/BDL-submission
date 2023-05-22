from Utils.utils import *
from Utils.evaluation import *
from Utils.dataset import BDL_dataset, Benchmark_test_dataset
from Models.models import BDL, Discriminator

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
import argparse

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


#%% Configuration

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--dim', type=int, default=64)

parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=1.5)
parser.add_argument('--x', type=float, default=0.01)

opt = parser.parse_args()

gpu = torch.device('cuda:' + str(opt.gpu))

lr = opt.lr
weight_decay = opt.weight_decay
dim = opt.dim

alpha = opt.alpha
beta = opt.beta
x = opt.x

tau = 0.999
max_epoch = 500

#%% Data load & preparation
## Load dataset & biased knowledge with affinity
user_num, item_num, train_mat, train_records, valid_mat, test_mat = read_data('./Resources/', 'yahoo')

b_mat = torch.load('./Resources/biased_knowledge/yahoo_bias_affinity').to(gpu)
biased_knowledge = torch.load('./Resources/biased_knowledge/yahoo_biased_teacher').to(gpu)

## we only use the biasd knowledge for high-affinity pairs
train_records = torch.LongTensor(train_records)
b_mat[train_records[:,0], train_records[:,1]] = b_mat.min()
b_user, b_item = torch.where(b_mat > torch.quantile(b_mat.view(-1), 1-x))

S_01 = []
for i in range(b_user.size(0)):
    u_id, i_id = int(b_user[i]), int(b_item[i])
    S_01.append((u_id, i_id, float(biased_knowledge[u_id][i_id])))
    
S_01_mat = list_to_dict(S_01)

train_dataset = BDL_dataset(user_num, item_num, train_mat, train_records, b_mat, S_01_mat, S_01)
train_loader = data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_dataset = Benchmark_test_dataset(user_num, item_num, train_mat, valid_mat, test_mat, gpu)


#%% Model & Optimizer
model = BDL(user_num, item_num, dim, tau, gpu).to(gpu)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

model_d = Discriminator(dim).to(gpu)
optimizer_d = optim.Adam(model_d.parameters(), lr=lr, weight_decay=weight_decay)

model_path = None

#%% Train & evaluation
train_loss_list = []

best_valid_MSE = 999
best_val_result, test_result = 1, 1
early_stopping = 0

for epoch in range(max_epoch):

    train_loader.dataset.dataset_construction()
    
    reliability_mat = model.compute_reliability()
    u, i = train_loader.dataset.train_arr[:,0].type(torch.LongTensor), train_loader.dataset.train_arr[:,2].type(torch.LongTensor)
    reliability_arr = reliability_mat[u, i]
    
    epoch_loss = []

    for mini_batch in train_loader:
        mini_batch['user'] = mini_batch['user'].type(torch.LongTensor).to(gpu)
        mini_batch['item_b'] = mini_batch['item_b'].type(torch.LongTensor).to(gpu)
        mini_batch['item_ub'] = mini_batch['item_ub'].type(torch.LongTensor).to(gpu)
        mini_batch['rating'] = mini_batch['rating'].to(gpu).unsqueeze(-1)
        
        ## Forward Pass
        model.train()
        model_d.freeze_clf()
        optimizer.zero_grad()
        
        ### Backbone model
        h_ui, h_uj = model.forward_triple(mini_batch['user'], mini_batch['item_b'], mini_batch['item_ub'])
        CE_loss = model.get_preference_loss(h_ui, mini_batch['rating'])
        
        ### Debiasing by distribution alignment
        batch_user = torch.cat([mini_batch['user'], mini_batch['user']], 0)
        batch_item = torch.cat([mini_batch['item_b'], mini_batch['item_ub']], 0)
        batch_h = torch.cat([h_ui, h_uj], 0)
        batch_group = train_loader.dataset.get_affinity_group(batch_user, batch_item)
        adv_loss = model_d.get_adv_loss(batch_h, batch_group)
        
        ### Debiasing by bias-guided preference learning
        KD_loss = model.get_KD_loss(h_ui, mini_batch['rating'])
        r_filter = (reliability_mat[mini_batch['user'], mini_batch['item_ub']] * 1.).unsqueeze(-1)
        conf_loss = model.get_conf_with_filter((h_ui, h_uj), r_filter)
        
        batch_loss = CE_loss - alpha * adv_loss +  beta * (KD_loss + conf_loss)
    
        ## Backward and optimize
        batch_loss.backward()
        optimizer.step()
        epoch_loss.append(batch_loss)
          
        ## discriminator learning
        model_d.unfreeze_clf()
        optimizer_d.zero_grad()
        disc_loss = model_d.get_adv_loss(batch_h.detach(), batch_group)
        disc_loss.backward()
        optimizer_d.step()
        
        ## temporal mean learning
        model._update_target()
        
    epoch_loss = torch.mean(torch.stack(epoch_loss)).data.cpu().numpy()
    train_loss_list.append(epoch_loss)

    if epoch % 1 == 0:
        improved = False

        model.eval()
        with torch.no_grad():
            eval_results = evaluate(model, gpu, test_dataset)

            if eval_results['valid']['MSE'] < best_valid_MSE:
                improved = True
                best_valid_MSE = eval_results['valid']['MSE']
                best_val_result = eval_results['valid']
                test_result = eval_results['test']
                early_stopping = 0
                
                if model_path != None:
                    torch.save(model.state_dict(), model_path)
            else:
                improved = False
                early_stopping += 1

            print_result(epoch, max_epoch, epoch_loss, eval_results, improved)

    if early_stopping >= 5:
        break

print()
print("Valid Results")
print('MSE:', round(best_val_result['MSE'], 4), ', MAE:', round(best_val_result['MAE'], 4), ', NDCG:', round(best_val_result['NDCG'], 4))

print("Test Results")
print('MSE:', round(test_result['MSE'], 4), ', MAE:', round(test_result['MAE'], 4), ', NDCG:', round(test_result['NDCG'], 4))

