from utils import setup_seed, arg_parse, visualization
from load_data import load_data
from train import train_dhl,train_gcn,train_mlp,train_gat
import json
import torch
from networks import HGNN_classifier, GCN, GAT, MLP
import torch.nn.functional as F
import time
import numpy as np
chosse_trainer = {
    'dhl':train_dhl,
    'gcn':train_gcn,
    'MLP':train_mlp,
    'gat':train_gat
}

args = arg_parse()
print(args.device)

data = load_data(args)

fts = data['fts']
lbls = data['lbls']

args.in_dim = fts.shape[1]
args.out_dim = lbls.max().item() + 1
args.min_num_edges = args.k_e

acc_list = []

for i in range(10):
    print(f"\n========== 第 {i + 1} 次训练 ==========")
    setup_seed(args.seed)
    best_acc = chosse_trainer[args.model](data, args)
    print(f"第 {i + 1} 次测试准确率：{best_acc:.4f}")
    acc_list.append(best_acc)

test_acc_array = np.array(acc_list)
print(acc_list)
avg_acc = test_acc_array.mean()
std_acc = test_acc_array.std()
# args_list.append(args.__dict__)
print("\n========================================")
print(f"10 次测试准确率平均值：{avg_acc:.4f}")
print(f"标准差（std）：{std_acc:.4f}")
print("========================================\n")

# ############################################## visualization
# chosse_model = {
#     'dhl':HGNN_classifier,
#     'gcn':GCN,
#     'MLP':MLP,
#     'gat':GAT
# }
#
# model = chosse_model[args.model](args)
# state_dict = torch.load('model.pth',map_location=args.device)
# model.load_state_dict(state_dict)
# model.to(args.device)
#
#
# model.eval()
# mask = data['test_idx']
# labels = data['lbls'][mask]
#
# out, x, H, H_raw,edges = model(data,args)
# pred = F.log_softmax(out, dim=1)
#
# _, pred = pred[mask].max(dim=1)
# correct = int(pred.eq(labels).sum().item())
# acc = correct / len(labels)
#
# print("test Acc ===============> ", acc)
#
# visualization(model, data, args, title=None)
# # with open('commandline_args{}.txt'.format(args.cuda), 'w') as f:
# #     json.dump([args.__dict__,args.__dict__], f, indent=2)