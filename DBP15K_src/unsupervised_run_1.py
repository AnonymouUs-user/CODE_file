import torch
import gc
import random
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataset import DBPDataset
import argparse
import parser
from util import *
from datetime import datetime
from torch.optim.lr_scheduler import ExponentialLR
from model import *
import math


def seed_torch(seed=1029):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True  # 选择确定性算法

current_date = datetime.now().strftime("%m%d_%H%M")
print(current_date)
max_hit0 = 0
max_hit1_all = 0

class RUN():
    def __init__(self):

        self.pse_pair = None
        self.pse_train_pair = None
        self.args = self.parse_options(argparse.ArgumentParser())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G_dataset = None
        self.side_modalities = None

        self.dev_pair = None
        self.exp_mod = True
        # self.exp_mod = False
        self.semi_pred = True
        self.confidence = True

        self.L = 0
        self.csp = 1
        # self.csp = 2

        # zh-en
        # self.ST_weight = 2.5
        # self.N_weight = 1.8
        # self.V_weight = 1.3
        # self.A_weight = 2.5

        # ja-en
        # self.ST_weight = 2.45
        # self.N_weight = 2.5
        # self.V_weight = 1.5
        # self.A_weight = 2

        self.ST_weight = self.args.ST
        self.N_weight = self.args.N
        self.V_weight = self.args.V
        self.A_weight = self.args.A

        print('ST_weight:{}'.format(self.ST_weight))
        print('N_weight:{}'.format(self.N_weight))
        print('V_weight:{}'.format(self.V_weight))
        print('A_weight:{}'.format(self.A_weight))

        self.sig = self.args.sig
        print("sig is {}".format(self.sig))

        self.u = self.args.u
        # self.u = 0.8
        # self.u = 0.9
        # self.u = 0.85

        self.lr = 0.005
        # self.droprate = 0.3
        # self.droprate = 0.2
        # self.droprate = 0.4
        self.droprate = self.args.droprate
        # self.droprate = 0.6
        # self.droprate = 0.65
        self.trainset_shuffle = True
        # self.trainset_shuffle = False

        # self.train_epoch = 70
        self.train_epoch = 100
        # self.train_epoch = 50
        # self.batchsize = 512
        # self.batchsize = 400
        self.batchsize = 1024
        # self.batchsize = 1024

        self.hiden_size = 128
        # self.hiden_size = 300

        self.thred = 0.99
        # self.thred = 0.999
        # self.thred = 0.95
        # self.pre_thred = 0.99999
        # self.pre_thred = 0.999
        self.pre_thred = 0.99
        # self.pre_thred = 0.9999
        # self.pre_thred = 0.98
        # self.pre_thred = 0.9
        # self.pre_thred = 0.88
        # self.pre_thred = 0.75
        # self.pre_thred = 0.7
        # self.pre_thred = 0.6
        # self.pre_thred = 0.8
        # self.pre_thred = 0.9

        self.thred_weight = 0.95
        # self.thred_weight = 1
        # self.thred_weight = 0.99
        # self.thred_weight = 0.9
        # self.thred_weight = 0.85

        self.side_weight = 1
        self.side_weight_rate = 1

        self.remove_rest_set_1 = set()
        self.remove_rest_set_2 = set()
        print("batchsize is {}".format(self.batchsize))

    @staticmethod
    def parse_options(parser):
        parser.add_argument('dataset', type=str, default='zh-en', help='dataset name')
        parser.add_argument('unsupervised_modality', type= str, default='V', help='V or N')
        parser.add_argument('sig', type=float, default='0.5', help='sig num')
        parser.add_argument('droprate', type=float, default='0.3', help='droprate num')
        parser.add_argument('ST', type=float, default='0.3', help='ST num')
        parser.add_argument('N', type=float, default='0.3', help='ST num')
        parser.add_argument('V', type=float, default='0.3', help='ST num')
        parser.add_argument('A', type=float, default='0.3', help='ST num')
        parser.add_argument('u', type=float, default='0.3', help='ST num')


        return parser.parse_args()

    def load_dataset(self):
        print('Loading dataset...')
        dataset = self.args.dataset
        with open("../results/{}_{}_{}_sig{}.txt".format(dataset, current_date, self.args.unsupervised_modality, str(self.sig)), 'a') as f:
            f.write(dataset + '_' + current_date + ':\n')
            f.write('sig: ' + str(self.sig) + '\n')
            f.write('droprate: ' + str(self.droprate) + '\n')

        self.G_dataset = DBPDataset('../data/DBP15K/DBP_1/{}'.format(dataset),
                                    device=self.device)
        # load test_pair
        self.dev_pair = self.G_dataset.test_pair
        self.dev_pair_gold = self.dev_pair.copy()
        self.dev_pair_1, self.dev_pair_2 = self.dev_pair.copy().T
        self.dev_pair_1 = list(self.dev_pair_1)
        self.dev_pair_2 = list(self.dev_pair_2)

        self.dev_pair = torch.tensor(self.dev_pair)
        print("dev set: " + str(len(self.dev_pair)))

        # side_modality_load
        self.side_modalities = {}
        ECS_file_path =[]
        for filename in os.listdir('../data/DBP15K/DBP_ECS_1'):
            if 'Att' in filename:
                if dataset in filename and '_3' in filename:
                    ECS_file_path.append('../data/DBP15K/DBP_ECS_1/' + filename)
            elif dataset in filename:
                ECS_file_path.append('../data/DBP15K/DBP_ECS_1/' + filename)
            else:
                continue
        for filename in ECS_file_path:
            if filename.endswith('.npy'):
                moda_np = np.load(filename)
                moda_np = np.exp(moda_np)
                ecs_name = filename.split('/')[-1].split('.')[0]
                print(ecs_name)
                if 'Name' in filename:
                    moda_np = (moda_np - moda_np.min()) / (moda_np.max() - moda_np.min())
                    self.side_modalities[ecs_name] = moda_np
                else:
                    if 'Vis' in filename:
                        # moda_np = np.exp
                        moda_np = (moda_np - moda_np.min()) / (moda_np.max() - moda_np.min())
                        self.side_modalities[ecs_name] = moda_np
                    if 'Att' in filename:
                        moda_np = (moda_np - moda_np.min()) / (moda_np.max() - moda_np.min())
                        self.side_modalities[ecs_name] = moda_np

        # generate pse_train_pair based on dev_pair
        unsup_mod = self.args.unsupervised_modality
        if unsup_mod == 'V':
            unsup_file_name = 'Vis_' + dataset
        elif unsup_mod == 'N':
            unsup_file_name = 'Name_' + dataset
        else:
            print('Unsupervised modality should be V or N')

        self.pse_train_pair = self.pred_pair_generate(self.side_modalities[unsup_file_name])
        # self.pse_train_pair = self.pred_pair_generate_2(self.side_modalities[unsup_file_name])

        self.pse_pair = self.pse_train_pair.copy()
        check_pairs_set = {(a, b) for a, b, c in self.pse_pair}
        pair_true_num, pair_accuracy = self.pse_pair_accuacy(check_pairs_set, self.dev_pair_gold)
        print("pair_true_num: {}".format(pair_true_num))
        print('pse pair accuracy: ' + str(pair_accuracy))
        self.pse_train_pair = torch.tensor(list(self.pse_train_pair)).to(self.device)
        # exit(0)

        print("pse train set: " + str(len(self.pse_train_pair)))

        # generate rest_set based on pse_train_pair
        self.rest_set_1 = self.dev_pair_1
        self.rest_set_2 = self.dev_pair_2
        self.rows_to_keep = self.rest_set_1.copy()
        self.cols_to_keep = self.rest_set_2.copy()
        self.r_len = len(self.rows_to_keep)
        self.c_len = len(self.cols_to_keep)

        self.ent_adj, self.rel_adj, \
        self.node_size, self.rel_size, \
        self.adj_list, self.r_index, \
        self.r_val, self.triple_size, self.adj = self.G_dataset.reconstruct_search(None,
                                                                                    None,
                                                                                    self.G_dataset.kg1,
                                                                                    self.G_dataset.kg2,
                                                                                    new=True)

    def pred_pair_generate(self, score):
        pred_set = set()
        A = score.argmax(axis=0)
        B = score.argmax(axis=1)
        cont = 0
        for i, j in enumerate(A):
            if B[j] == i:
                cont += 1
                sc = (score[j][i] + score[i][j]) / 2
                if sc > self.sig:
                    new_conf = 1
                else:
                    new_conf = math.exp(-0.5 * self.u * (sc - self.sig) ** 2)
                if new_conf > self.pre_thred:
                    pred_set.add((self.dev_pair_1[j], self.dev_pair_2[i], new_conf))
        # print(cont)
        # exit(0)
        return pred_set

    def pred_pair_generate_2(self, score):
        topk = 100
        # topk = 1000
        # topk = 1500
        # topk = 2000
        # topk = 3000
        # topk = 3500
        # topk = 4000
        # topk = 5000
        def get_topk_indices(M, K=1000):
            H, W = M.shape
            M_view = M.view(-1)
            # M_view = M.view()
            vals, indices = M_view.topk(K)
            two_d_indices = torch.cat(((indices // W).unsqueeze(1), (indices % W).unsqueeze(1)), dim=1)
            return two_d_indices

        visual_links = set()
        used_inds = []
        count = 0
        score =torch.tensor(score)
        two_d_indices = get_topk_indices(score, topk * 100)
        for ind in two_d_indices:
            if self.dev_pair_1[ind[0]] in used_inds:
                continue
            if self.dev_pair_2[ind[1]] in used_inds:
                continue
            used_inds.append(self.dev_pair_1[ind[0]])
            used_inds.append(self.dev_pair_2[ind[1]])

            # visual_links.add((self.dev_pair_1[ind[0]], self.dev_pair_2[ind[1]], score[ind[0]][ind[1]].item()))

            sc = (score[ind[0]][ind[1]].item() + score[ind[1]][ind[0]].item()) / 2
            if sc > self.thred:
                new_conf = 1
            else:
                new_conf = math.exp(-0.5 * self.u * (sc - self.sig) ** 2)
            visual_links.add((self.dev_pair_1[ind[0]], self.dev_pair_2[ind[1]], new_conf))

            count += 1
            if count == topk:
                break
        # print(count)
        # exit(0)

        return visual_links


    def init_model(self):
        # self.depth = 2
        self.depth = 3
        # self.depth = 4

        self.structure_encoder = ST_Encoder_Module(
            node_hidden=self.hiden_size,
            rel_hidden=self.hiden_size,
            node_size=self.node_size,
            rel_size=self.rel_size,
            device=self.device,
            dropout_rate=self.droprate,
            depth=self.depth).to(self.device)

        self.loss_model = Loss_Module(node_size=self.node_size, gamma=2).to(self.device)

        # self.optimizer = torch.optim.RMSprop(self.structure_encoder.parameters(), lr=self.lr)

        # self.optimizer = torch.optim.Adam(self.structure_encoder.parameters(), lr=self.lr,
        #                                   betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        self.optimizer = torch.optim.AdamW(self.structure_encoder.parameters(),
                                           lr=self.lr, weight_decay=0)

        self.scheduler = ExponentialLR(self.optimizer, gamma=0.999)

        # total_params = sum(p.numel() for p in self.structure_encoder.parameters() if p.requires_grad)
        # print(total_params)

    def run(self):
        self.load_dataset()
        self.init_model()
        train_epoch = self.train_epoch
        batch_size = self.batchsize
        for epoch in range(train_epoch):
            print("now is epoch " + str(epoch))
            self.structure_encoder.train()
            if self.trainset_shuffle:
                num_rows = self.pse_train_pair.size(0)
                random_indices = torch.randperm(num_rows)
                self.pse_train_pair = self.pse_train_pair[random_indices]

            for i in range(0, len(self.pse_train_pair), batch_size):
                batch_pair = self.pse_train_pair[i:i + batch_size]
                if len(batch_pair) == 0:
                    continue
                feature_list = self.structure_encoder(
                    self.ent_adj, self.rel_adj, self.node_size,
                    self.rel_size, self.adj_list, self.r_index, self.r_val,
                    self.triple_size, mask=None)

                loss = self.loss_model(batch_pair, feature_list[0], weight=True)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
            # loss.backward()

            # if epoch % 5 == 4:
            if epoch % 2 == 0 and epoch >= 4:
            # if epoch % 8 == 0 and epoch >= 4:
            # if epoch % 4 == 0 and epoch >= 4:
                gid1, gid2 = self.dev_pair.T
                print(len(gid1))
                self.structure_encoder.eval()
                with torch.no_grad():
                    feature_list = self.structure_encoder(
                        self.ent_adj.to(self.device), self.rel_adj.to(self.device),
                        self.node_size, self.rel_size,
                        self.adj_list.to(self.device),
                        self.r_index.to(self.device), self.r_val.to(self.device),
                        self.triple_size,
                        mask=None)

                    out_feature = feature_list[0].cpu()
                    del feature_list
                    torch.cuda.empty_cache()

                    out_feature = out_feature / (torch.linalg.norm(out_feature, dim=-1, keepdim=True) + 1e-5)
                    index_a = torch.LongTensor(gid1)
                    index_b = torch.LongTensor(gid2)

                    Lvec = out_feature[index_a]
                    Rvec = out_feature[index_b]
                    Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
                    Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)

                    print("*********************************************\n")
                    result = self.sinkhorn_ST_test(Lvec, Rvec, device=self.device,
                                                            len_dev=len(self.dev_pair))
                    print("*********************************************\n")

                    hits0 = float(result['hits@1'])
                    global max_hit0
                    global max_hit1_all
                    if hits0 > max_hit0:
                        max_hit0 = hits0
                        # if epoch == 24:
                        #     print("continue")
                        with open("../results/{}_{}_{}_sig{}.txt".format(self.args.dataset, current_date, self.args.unsupervised_modality, str(self.sig)), 'a') as f:
                            f.write(str(epoch)+':\n')
                            f.write(str(result) + '\n')
                        if hits0 > 80:
                            print("=============================================\n")
                            all_result = self.sinkhorn_test(Lvec, Rvec, device=self.device,len_dev=len(self.dev_pair))
                            if float(all_result['hits@1']) > max_hit1_all:
                                max_hit1_all = float(all_result['hits@1'])

                            print("=============================================\n")
                            with open("../results/{}_{}_{}_sig{}.txt".format(self.args.dataset, current_date, self.args.unsupervised_modality, str(self.sig)), 'a') as f:
                                f.write(str(epoch)+':\n')
                                f.write(str(all_result) + '\n')

                    else:
                        all_result = self.sinkhorn_test(Lvec, Rvec, device=self.device, len_dev=len(self.dev_pair))
                        if float(all_result['hits@1']) > max_hit1_all:
                            max_hit1_all = float(all_result['hits@1'])
                        else:
                            print("max_hit0={}".format(max_hit0))
                            print("max_hit1_all={}".format(max_hit1_all))
                            with open("../results/{}_{}_{}_sig{}.txt".format(self.args.dataset, current_date,
                                                                             self.args.unsupervised_modality,
                                                                             str(self.sig)), 'a') as f:
                                f.write("max_hit0={}".format(max_hit0) + ':\n')
                                f.write("max_hit1_all={}".format(max_hit1_all) + '\n')
                            exit(0)
                        # print('Over!')
                        # exit(0)

            # if epoch % 5 == 4 and self.semi_pred:
            if epoch >= 4 and epoch % 2 == 0 and self.semi_pred:
            # if epoch >= 4 and epoch % 8 == 0 and self.semi_pred:
                self.structure_encoder.eval()
                with torch.no_grad():
                    gid1 = torch.tensor(np.array(self.rest_set_1))
                    gid2 = torch.tensor(np.array(self.rest_set_2))
                    # print('rest set shape is : {} {}'.format(len(gid1), len(gid2)))
                    # out_feature = feature_list[0].cpu()
                    Lvec = out_feature[gid1]
                    Rvec = out_feature[gid2]
                    Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
                    Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)

                    scores = self.sim_results(Lvec, Rvec)
                    rows_to_keep = [i for i in range(self.r_len) if i not in self.remove_rest_set_1]
                    cols_to_keep = [j for j in range(self.c_len) if j not in self.remove_rest_set_2]

                    # print("self.remove_rest_set shape: set1:{} set2:{}".format(len(self.remove_rest_set_1),
                    #                                                            len(self.remove_rest_set_2)))
                    if len(rows_to_keep) != len(cols_to_keep):
                        print('wrong: len(rows_to_keep) != len(cols_to_keep)')
                        print(len(rows_to_keep))
                        print(len(cols_to_keep))
                        exit(0)
                    if scores.shape[0] != len(rows_to_keep):
                        print('wrong: scores.shape[0] != len(rows_to_keep) ')
                        exit(0)
                    assert scores.shape[0] == len(rows_to_keep) and len(rows_to_keep) == len(cols_to_keep)

                    new_pair = set()
                    new_pair.update(self.pred_pair_confidence(scores))

                    # 结合所有模态生成pair

                    for i, (name, side_score) in enumerate(self.side_modalities.items()):
                        if 'Vis' in name:
                            self.side_weight = self.V_weight
                        elif 'N' in name:
                            self.side_weight = self.N_weight
                        else:
                            self.side_weight = self.A_weight
                            # self.side_weight = 0
                        # self.side_weight = 1

                        # 所有模态求和，生成pair
                        # scores += torch.Tensor(side_score[np.ix_(rows_to_keep,cols_to_keep)]) * self.side_weight
                        # new_pair.update(self.pred_pair_confidence(scores / float(i + 2)))

                        # 每个模态都单独生成pair
                        s = torch.Tensor(side_score[np.ix_(rows_to_keep,cols_to_keep)] * self.side_weight)
                        new_pair.update(self.pred_pair_confidence(s))

                    print('new_pair len: {}'.format(len(new_pair)))
                    if self.csp == 1:
                        new_pair = self.delete_repeat_pair(new_pair)
                    else:
                        new_pair = self.choose_repeat_pair(new_pair)

                    print('after csp, new_pair len:{}'.format(len(new_pair)))

                    self.pse_pair.update(new_pair)

                    print('pse_pair len: {}'.format(len(self.pse_pair)))
                    # 对伪标签集中的标签也进行筛选
                    if self.csp == 1:
                        self.pse_pair = self.delete_repeat_pair(self.pse_pair)
                    else:
                        self.pse_pair = self.choose_repeat_pair(self.pse_pair)

                    check_pairs_set = {(a, b) for a, b, c in self.pse_pair}
                    pair_true_num, pair_accuracy = self.pse_pair_accuacy(check_pairs_set, self.dev_pair_gold)


                    self.pse_train_pair = torch.tensor(list(self.pse_pair.copy())).to(self.device)


                    with open("../results/{}_{}_{}_sig{}.txt".format(self.args.dataset, current_date,
                                                               self.args.unsupervised_modality, str(self.sig)), 'a') as f:
                        f.write(str(epoch) + ':\n')
                        f.write('pair true num: {}'.format(pair_true_num) + ':\n')
                        f.write('pair accuracy: {}'.format(pair_accuracy) + ':\n')
                        f.write('pse train pair len: {}'.format(len(self.pse_train_pair))+ '\n')
                        f.write('increase pseudo pair: {}'.format(len(new_pair)) + '\n')

                    print("*********************************************\n")
                    print("pair_true_num:{}".format(pair_true_num))
                    print('pair accuracy: {}'.format(pair_accuracy))
                    print('pse train pair len: {}'.format(len(self.pse_train_pair)))
                    print("increase pseudo pair: {}".format(len(new_pair)))
                    print("*********************************************\n")

                    count = 0
                    for (e1, e2, conf) in new_pair:
                        if e1 in self.rest_set_1 and e2 in self.rest_set_2:
                            try:
                                if e1 in self.rows_to_keep and e2 in self.cols_to_keep:
                                    index_1 = self.rows_to_keep.index(e1)
                                    index_2 = self.cols_to_keep.index(e2)
                            except ValueError:
                                print(f"元素 {e1} 或 {e2} 不在rows_to_keep, cols_to_keep集合中。")

                            if index_1 in self.remove_rest_set_1 or index_2 in self.remove_rest_set_2:
                                print(index_1)
                                print(index_1 in self.remove_rest_set_1)
                                print(index_2)
                                print(index_2 in self.remove_rest_set_2)
                                continue
                            else:
                                self.remove_rest_set_1.add(index_1)
                                self.remove_rest_set_2.add(index_2)
                                count = count + 1

                            try:
                                if e1 in self.rest_set_1 and e2 in self.rest_set_2:
                                    self.rest_set_1.remove(e1)
                                    self.rest_set_2.remove(e2)

                            except ValueError:
                                print(f"元素 {e1} 或 {e2} 不在rest_set_1, rest_set_2。")

                    print("number of new_pair is {}, real remove number is {}".format(len(new_pair), count))
                    self.thred = self.thred * self.thred_weight
                    self.side_weight = self.side_weight * self.side_weight_rate


    def sinkhorn_ST_test(self, sourceVec, targetVec, device, len_dev):
        sim_mat = self.sim_results(sourceVec, targetVec)
        sim_mat = (sim_mat - sim_mat.min()) / (sim_mat.max() - sim_mat.min())
        sim_mat = torch.exp(sim_mat)

        sim_mat = sim_mat * self.ST_weight
        if self.L == 0:
            sim_mat = sim_mat.T

        if self.L == 1:
            gid1, gid2 = self.dev_pair.T
            new_adj_1 = self.adj[gid1, :][:, gid1]
            new_adj_2 = self.adj[gid2, :][:, gid2]
            new_adj_1 = new_adj_1 / (np.linalg.norm(new_adj_1, axis=-1, keepdims=True) + 1e-5)
            new_adj_2 = new_adj_2 / (np.linalg.norm(new_adj_2, axis=-1, keepdims=True) + 1e-5)
            new_adj_1 = torch.FloatTensor(new_adj_1)
            new_adj_2 = torch.FloatTensor(new_adj_2)
            sim_mat = sim_mat.T + new_adj_2 * sim_mat.T * new_adj_1.T

        sim_mat_r = 1 - sim_mat

        # matrix_sinkhorn
        if sim_mat_r.dim == 3:
            M = sim_mat_r
        else:
            M = sim_mat_r.view(1, sim_mat_r.size(0), -1)
        M = M.to(device)
        m, n = sim_mat_r.shape
        a = torch.ones([1, m], requires_grad=False, device=device)
        b = torch.ones([1, n], requires_grad=False, device=device)
        P = sinkhorn(a, b, M, 0.02, max_iters=100, stop_thresh=1e-3)
        P = view2(P)
        del M, a, b
        torch.cuda.empty_cache()

        # evaluate_sim
        result = evaluate_sim_matrix(link=torch.stack([torch.arange(len_dev),
                                                       torch.arange(len_dev)], dim=0),
                                     sim_x2y=P,
                                     no_csls=True)
        return result

    def sinkhorn_test(self, sourceVec, targetVec, device, len_dev):
        sim_mat = self.sim_results(sourceVec, targetVec)
        sim_mat = (sim_mat - sim_mat.min()) / (sim_mat.max() - sim_mat.min())
        sim_mat = torch.exp(sim_mat)

        sim_mat = sim_mat * self.ST_weight
        ST_mat = sim_mat
        count = 0
        if self.exp_mod:
            for name, side_score in self.side_modalities.items():
                if 'Vis' in name:
                    weight = self.V_weight
                elif 'N' in name:
                    weight = self.N_weight
                else:
                    weight = self.A_weight
                # side_score = np.exp(side_score)
                sim_mat += torch.Tensor(side_score * weight)
                count = count + 1
            print(count)

        sim_mat = sim_mat / float(count + 1)

        if self.L == 0:
            sim_mat = sim_mat.T

        if self.L == 1:
            gid1, gid2 = self.dev_pair.T
            new_adj_1 = self.adj[gid1, :][:, gid1]
            new_adj_2 = self.adj[gid2, :][:, gid2]
            new_adj_1 = new_adj_1 / (np.linalg.norm(new_adj_1, axis=-1, keepdims=True) + 1e-5)
            new_adj_2 = new_adj_2 / (np.linalg.norm(new_adj_2, axis=-1, keepdims=True) + 1e-5)
            new_adj_1 = torch.FloatTensor(new_adj_1)
            new_adj_2 = torch.FloatTensor(new_adj_2)
            sim_mat = sim_mat.T + new_adj_2 * sim_mat.T * new_adj_1.T

        sim_mat_r = 1 - sim_mat

        # matrix_sinkhorn
        if sim_mat_r.dim == 3:
            M = sim_mat_r
        else:
            M = sim_mat_r.view(1, sim_mat_r.size(0), -1)
        M = M.to(device)
        m, n = sim_mat_r.shape
        a = torch.ones([1, m], requires_grad=False, device=device)
        b = torch.ones([1, n], requires_grad=False, device=device)
        P = sinkhorn(a, b, M, 0.02, max_iters=100, stop_thresh=1e-3)
        P = view2(P)

        del M, a, b
        torch.cuda.empty_cache()

        # evaluate_sim
        result = evaluate_sim_matrix(link=torch.stack([torch.arange(len_dev),
                                                       torch.arange(len_dev)], dim=0),
                                     sim_x2y=P,
                                     no_csls=True)
        return result

    def sim_results(self, Matrix_A, Matrix_B):
        # A x B.t
        A_sim = torch.mm(Matrix_A, Matrix_B.t())
        return A_sim

    def pred_pair_confidence(self, score):

        new_set = set()
        A = score.argmax(axis=0)
        B = score.argmax(axis=1)
        for i, j in enumerate(A):
            # if B[j] == i and (score[j][i] > self.sig or score[i][j] > self.thred):
            #     if score[j][i] < self.thred or score[i][j] < self.thred:
            if B[j] == i and (score[j][i] > self.sig or score[i][j] > self.sig):
                if score[j][i] < self.sig or score[i][j] < self.sig:
                    sc = (score[j][i] + score[i][j]) / 2
                    new_conf = math.exp(-0.5 * self.u * (sc - self.sig) ** 2)
                else:
                    new_conf = 1
                if self.confidence and new_conf > self.thred :
                    new_set.add((self.rest_set_1[j], self.rest_set_2[i], new_conf))
                else:
                    new_set.add((self.rest_set_1[j], self.rest_set_2[i], 1))

        return new_set

    def delete_repeat_pair(self, pair):
        a_to_bs = {}
        b_to_as = {}
        for a, b, conf in pair:
            if a in a_to_bs:
                a_to_bs[a].append(b)
            else:
                a_to_bs[a] = [b]
            if b in b_to_as:
                b_to_as[b].append(a)
            else:
                b_to_as[b] = [a]

        conflicting_tuples = set()
        for a, bs in a_to_bs.items():
            if len(bs) > 1:
                for b in bs:
                    conflicting_tuples.add((a, b))
        for b, a_s in b_to_as.items():
            if len(a_s) > 1:
                for a in a_s:
                    conflicting_tuples.add((a, b))

        print("conflicting_tuples: {}".format(len(conflicting_tuples)))

        conflicting_tuples_v = set()
        for (x, y) in conflicting_tuples:
            for triple in pair:
                if triple[0] == x and triple[1] == y:
                    # print(triple)
                    conflicting_tuples_v.add((triple[0], triple[1], triple[2]))
                    # print(conflicting_tuples_v)

        new_pair = pair.difference(conflicting_tuples_v)
        return new_pair

    def delete_repeat_pair_tensor(self, pair_tensor):
        a_to_bs = {}
        b_to_as = {}
        for a, b, conf in pair_tensor:
            if a in a_to_bs:
                a_to_bs[a].append(b)
            else:
                a_to_bs[a] = [b]
            if b in b_to_as:
                b_to_as[b].append(a)
            else:
                b_to_as[b] = [a]

        conflicting_tuples = set()
        for a, bs in a_to_bs.items():
            if len(bs) > 1:
                for b in bs:
                    conflicting_tuples.add((a, b))
        for b, a_s in b_to_as.items():
            if len(a_s) > 1:
                for a in a_s:
                    conflicting_tuples.add((a, b))
        if len(conflicting_tuples) > 0:
            new_pair = pair_tensor[~torch.isin(pair_tensor[:, 0], torch.tensor(list(conflicting_tuples), device=pair_tensor.device)[:, 0])]
            return new_pair
        else:
            return pair_tensor

    def choose_repeat_pair(self, pair):
        max_triplets = {}
        for triplet in pair:
            x, y, z = triplet
            if x not in max_triplets or max_triplets[x][2] < z:
                max_triplets[x] = triplet
            if y not in max_triplets or max_triplets[y][2] < z:
                max_triplets[y] = triplet

        new_pair = set()
        for triplet in max_triplets.values():
            x, y, z = triplet
            if (x not in max_triplets or max_triplets[x] == triplet) and (
                    y not in max_triplets or max_triplets[y] == triplet):
                new_pair.add(triplet)
        print('pair len:{}'.format(len(pair)))
        print('max triple len:{}'.format(len(max_triplets)))
        print('new pair len:{}'.format(len(new_pair)))
        return new_pair

    def choose_repeat_pair_tensor(self, pair_tensor):
        max_triplets = {}
        for triplet in pair_tensor:
            x, y, z = triplet
            if x not in max_triplets or max_triplets[x][2] < z:
                max_triplets[x] = triplet
            if y not in max_triplets or max_triplets[y][2] < z:
                max_triplets[y] = triplet

        new_pair = pair_tensor[~torch.isin(pair_tensor[:, 0], torch.tensor(list(max_triplets.values()), device=pair_tensor.device)[:, 0])]
        return new_pair

    def pse_pair_accuacy(self, pse_pair, pair_gold):
        pair_gold = {tuple(row) for row in pair_gold}
        return len(pse_pair.intersection(pair_gold)), len(pse_pair.intersection(pair_gold)) / len(pse_pair)

    def pse_pair_confidence_accuaracy(self, pse_pair, pair_gold):
        pair_gold_confidence = {tuple(row, 1) for row in pair_gold}
        pass






if __name__ == "__main__":
    seed_torch()

    try:
        model = RUN()
        model.run()
    except KeyboardInterrupt:  # 捕获键盘中断异常
        print("手动中断训练...")
        torch.cuda.empty_cache()

        # 释放未使用的显存
        gc.collect()
        torch.cuda.empty_cache()
        # 显示当前显存使用情况
        print(f"显存使用情况: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
    finally:
        # 确保显存被正确释放
        torch.cuda.empty_cache()

