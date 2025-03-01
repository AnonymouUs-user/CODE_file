import numpy as np
import torch
import torch.nn.functional as F
from dataset import DBPDataset
import argparse
from model import ST_Encoder_Module, Loss_Module, Weight_train_2
import gc
import random
import os
from util import *
from datetime import datetime
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import math
import pytz

timezone = pytz.timezone('Asia/Shanghai')
current_date = datetime.now(timezone).strftime("%m%d_%H%M")
print(current_date)
max_hit0 = 0


def seed_torch(seed=1029):
    # print('set seed')
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True  # 选择确定性算法


def read_list(file):
    l = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            l.append((cur[0], cur[1]))
    return l


def sinkhorn_test_3(scores, len_pair, device):

    # L = 0
    scores = scores.T

    # L = 1
    # scores = scores.T + new_adj_2 * scores.T * new_adj_1.T

    scores = torch.Tensor(scores).to(device)
    sim_mat_r = 1 - scores

    # matrix_sinkhorn
    if sim_mat_r.dim == 3:
        M = sim_mat_r
    else:
        M = sim_mat_r.view(1, sim_mat_r.size(0), -1)
    m, n = sim_mat_r.shape
    a = torch.ones([1, m], requires_grad=False, device=device)
    b = torch.ones([1, n], requires_grad=False, device=device)
    P = sinkhorn(a, b, M, 0.02, max_iters=100, stop_thresh=1e-3)
    P = view2(P)

    # evaluate_sim
    result = evaluate_sim_matrix(link=torch.stack([torch.arange(len_pair),
                                                   torch.arange(len_pair)], dim=0),
                                 sim_x2y=P,
                                 no_csls=True)
    return result

def explicit_score_test(score, dev_pair, device):
    gid1, gid2 = dev_pair.T
    pair_score = score[np.ix_(gid1.tolist(), (gid2 - score.shape[0]).tolist())]
    pair_score = torch.Tensor(pair_score).to(device)
    sim_mat_r = 1 - pair_score
    if sim_mat_r.dim == 3:
        M = sim_mat_r
    else:
        M = sim_mat_r.view(1, sim_mat_r.size(0), -1)
    m, n = sim_mat_r.shape
    a = torch.ones([1, m], requires_grad=False, device=device)
    b = torch.ones([1, n], requires_grad=False, device=device)
    M = M.to(device)
    P = sinkhorn(a, b, M, 0.02, max_iters=100, stop_thresh=1e-3)
    P = view2(P)
    result = evaluate_sim_matrix(link=torch.stack([torch.arange(pair_score.shape[0]),
                                                   torch.arange(pair_score.shape[0])], dim=0),
                                 sim_x2y=P,
                                 no_csls=True)

    return result

class RUN():
    def __init__(self):
        super(RUN, self).__init__()
        self.semi_pred = True
        self.parser = argparse.ArgumentParser()
        self.args = self.parse_options(self.parser)

        self.csp = self.args.CSP
        self.N = self.args.Name
        self.sig = self.args.sig

        self.exp_mod = True
        self.Exwithout = None
        self.weight_model = None

        self.ST_weight = 2.45
        self.N_weight = 2.25
        self.V_weight = 1.5
        self.A_weight = 1

        self.L = 0
        # self.L = 1
        self.Imp = 'ST'
        self.out_str = ''
        if self.Imp:
            self.out_str = self.out_str + str(self.Imp)

        if self.exp_mod:
            self.out_str = self.out_str + '+V+N'

        if self.semi_pred:
            self.out_str = self.out_str + '+csp{}'.format(self.csp)

        print('L={}'.format(self.L))
        print(self.out_str)

        print("L:{}, semi_pred={}, "
              "side_mode={}, self.Imp ={}".format(self.L, self.semi_pred, self.exp_mod, self.Imp))

        self.loss_model = None
        self.structure_encoder = None

        self.confidence = True
        self.thred = 0.95
        # self.thred = 0.9
        self.thred_weight = 0.95

        self.img_feature = None

        self.remove_rest_set_1 = set()
        self.remove_rest_set_2 = set()
        self.rest_pair = None
        self.G_dataset = None
        self.getembedding = None
        self.triple_size = None
        self.r_val = None
        self.r_index = None
        self.adj_list = None
        self.rel_size = None
        self.node_size = None
        self.rel_adj = None
        self.ent_adj = None
        self.dev_pair = None
        self.train_pair = None
        self.train_pair_confidence = None

        self.device = 'cuda:0'

        # 3-10-4  Golden-setting!!
        self.train_epoch = 40
        self.batchsize = 1024  # db
        print("batchsize is {}".format(self.batchsize))

        # self.batchsize = 1024 # 可以加速训练，但是效果会差一个点

        self.lr = 0.005
        # self.droprate = 0.1
        self.droprate = 0.3

        self.side_weight = 1
        self.side_weight_rate = 1

        self.trainset_shuffle = True



    @staticmethod
    def parse_options(parser):
        parser.add_argument('dataset', type=str, help='which dataset, ja_en, fr_en, zh_en', default='ja_en')
        parser.add_argument('sig', type=float, default='0.5')
        parser.add_argument('CSP', type=float, default='1')
        parser.add_argument('Name', type=float,default='0')
        parser.add_argument('epoch_weight', type=float,default='0')

        return parser.parse_args()

    def load_dataset(self):
        print("1. load dataset....")

        dataset = self.args.dataset
        # creat file named datetime+dataset
        with open("../results/{}_{}_sig{}.txt".format(self.args.dataset, current_date, str(self.sig)), 'a') as f:
            f.write(dataset+':\n')
            f.write(str(self.out_str)+': \n')

        self.G_dataset = DBPDataset('../data/DBP15K/DBP_1/{}'.format(dataset),
                                    device=self.device)
        self.train_pair, self.dev_pair = self.G_dataset.train_pair, self.G_dataset.test_pair
        self.train_pair = torch.tensor(self.train_pair).to(self.device)
        self.dev_pair = torch.tensor(self.dev_pair)

        confidence_column = torch.ones(self.train_pair.shape[0], 1)
        self.train_pair_confidence = torch.cat((self.train_pair, confidence_column.to(self.device)), dim=1)

        self.rest_set_1 = self.G_dataset.rest_set_1
        self.rest_set_2 = self.G_dataset.rest_set_2

        self.rows_to_keep = self.G_dataset.rest_set_1.copy()
        self.cols_to_keep = self.G_dataset.rest_set_2.copy()
        self.r_len = len(self.rows_to_keep)
        self.c_len = len(self.cols_to_keep)

        print("train set: " + str(len(self.train_pair)))
        print("dev set: " + str(len(self.dev_pair)))
        self.true_num = len(self.train_pair)

        self.ent_adj, self.rel_adj, self.node_size, self.rel_size, \
        self.adj_list, self.r_index, self.r_val, self.triple_size, self.adj = self.G_dataset.reconstruct_search(None,
                                                                                                                None,
                                                                                                                self.G_dataset.kg1,
                                                                                                                self.G_dataset.kg2,
                                                                                                                new=True)
        # self.adj 2个KG联合的邻接矩阵

        print("     ent_adj size: " + str(self.ent_adj.shape) + "  device: " + str(self.ent_adj.device))
        print("     rel_adj size: " + str(self.rel_adj.shape))
        print("     node_size: " + str(self.node_size))
        print("     rel_size: " + str(self.rel_size))
        print("     adj_list size: " + str(len(self.adj_list)))
        print("     r_index size: " + str(self.r_index.shape))
        print("     r_val size: " + str(self.r_val.shape))
        print("     triple_size: " + str(self.triple_size))

        print("1. load dataset over....")

        self.side_modalities = {}
        ECS_file_path =[]
        self.modal_num = 1
        for filename in os.listdir('../data/DBP15K/DBP_ECS_1'):
            if dataset in filename:
                ECS_file_path.append('../data/DBP15K/DBP_ECS_1/' + filename)

        for filename in ECS_file_path:
            # print(filename)
            if 'Vis' in filename and 'new' in filename:
                self.w = self.V_weight
                print('new_vis')
            # elif 'attr' in filename and 'llmembed' in filename:
            elif 'Att' in filename:
                self.w = self.A_weight
                print('')
            # elif 'name' in filename and 'llmembed' in filename and self.N:
            elif 'Name' in filename and self.N:
                self.w = self.N_weight
                print('')
            else:
                # print('other')
                continue
            if filename.endswith('.npy'):
                moda_np = np.load(filename)
                # if moda_np.shape[0] == 15000:
                #     sinkhorn_test_3(moda_np, moda_np.shape[0], self.device)
                # else:
                #     explicit_score_test(moda_np,self.dev_pair, self.device)
                print(moda_np.shape)
                ecs_name = filename.split('/')[-1].split('.')[0]
                self.side_modalities[ecs_name] = moda_np * self.w
                print(ecs_name)
                self.modal_num += 1
        # exit(0)


    def init_model(self):
        # self.depth = 2
        self.depth = 4
        # self.depth = 5

        if self.Imp == 'ST':
            emb_dim = 128
            # emb_dim = 256
            self.structure_encoder = ST_Encoder_Module(
                node_hidden=emb_dim,
                rel_hidden=emb_dim,
                node_size=self.node_size,
                rel_size=self.rel_size,
                device=self.device,
                dropout_rate=self.droprate,
                depth=self.depth).to(self.device)

        self.loss_model = Loss_Module(node_size=self.node_size, gamma=2).to(self.device)

        self.optimizer = torch.optim.RMSprop(self.structure_encoder.parameters(), lr=self.lr)  # 3-9
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.999)

        total_params = sum(p.numel() for p in self.structure_encoder.parameters() if p.requires_grad)
        # print(total_params)

        self.weight_model = Weight_train_2(N_view=self.modal_num, device=self.device)
        self.optimizer_2 = torch.optim.RMSprop(self.weight_model.parameters(), lr=self.lr)
        self.scheduler_2 = ExponentialLR(self.optimizer_2, gamma=0.999)




    def run(self):
        self.load_dataset()
        self.init_model()

        trainset = []

        train_epoch = self.train_epoch
        batch_size = self.batchsize

        print("2. start training....\n")
        for epoch in range(train_epoch):
            print("now is epoch " + str(epoch))
            self.structure_encoder.train()
            self.weight_model.train()
            if self.trainset_shuffle:

                num_rows = self.train_pair_confidence.size(0)
                random_indices = torch.randperm(num_rows)
                self.train_pair_confidence = self.train_pair_confidence[random_indices]

            # for i in range(0, len(self.train_pair), batch_size):
            for i in range(0, len(self.train_pair_confidence), batch_size):
                # batch_pair = self.train_pair[i:i+batch_size]
                batch_pair = self.train_pair_confidence[i:i + batch_size]
                if len(batch_pair) == 0:
                    continue
                feature_list = self.structure_encoder(
                    self.ent_adj, self.rel_adj, self.node_size,
                    self.rel_size, self.adj_list, self.r_index, self.r_val,
                    self.triple_size, mask=None)

                loss = self.loss_model(batch_pair, feature_list[0], weight=True)
                if epoch > self.args.epoch_weight:
                    self.weight_t, cos_loss = self.weight_model(batch_pair.clone(),
                                                                feature_list[0].clone(), self.side_modalities)
                    cos_loss.backward(retain_graph=True)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            # test code
            if epoch % 2 == 0 and epoch >= 0:

                gid1, gid2 = self.dev_pair.T
                print(len(gid1))

                self.structure_encoder.eval()
                self.weight_model.eval()
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

                    # ST_result = self.sinkhorn_ST_test(Lvec, Rvec, device=self.device,
                    #                                len_dev=len(self.dev_pair))

                    # result = self.sinkhorn_test(Lvec, Rvec, device=self.device,
                    #                                         len_dev=len(self.dev_pair))
                    result = self.sinkhorn_test(Lvec, Rvec, device=self.device,
                                                len_dev=len(self.dev_pair), gid1=gid1, gid2=gid2)
                    hits0 = float(result['hits@1'])
                    global max_hit0
                    if hits0 > max_hit0:
                        max_hit0 = hits0
                        with open("../results/{}_{}_sig{}.txt".format(self.args.dataset, current_date, str(self.sig)), 'a') as f:
                            f.write(str(epoch)+':\n')
                            f.write(str(result) + '\n')
                    print("max_hit0={}".format(max_hit0))

            # if epoch >= 9 and epoch % 5 == 4 and self.semi_pred:
            if epoch >= 9 and epoch % 4 == 0 and self.semi_pred:
                # self.structure_encoder.eval()
                self.weight_model.eval()
                with torch.no_grad():

                    gid1 = torch.tensor(np.array(self.rest_set_1))
                    gid2 = torch.tensor(np.array(self.rest_set_2))
                    print('rest set shape is : {} {}'.format(len(gid1), len(gid2)))

                    Lvec = out_feature[gid1]
                    Rvec = out_feature[gid2]
                    Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
                    Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)

                    scores = self.sim_results(Lvec, Rvec)

                    # assert the number of rest set and pseduo-pair set
                    rows_to_keep = [i for i in range(self.r_len) if i not in self.remove_rest_set_1]
                    cols_to_keep = [j for j in range(self.c_len) if j not in self.remove_rest_set_2]
                    print("self.remove_rest_set shape: set1:{} set2:{}".format(len(self.remove_rest_set_1),
                                                                               len(self.remove_rest_set_2)))
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

                    for i, (name, side_score) in enumerate(self.side_modalities.items()):
                        if 'Att' in name or 'Name' in name:
                            s = torch.Tensor(side_score[np.ix_(rows_to_keep, cols_to_keep)] * self.side_weight)
                            new_pair.update(self.pred_pair_confidence(s))
                        else:
                            new_pair.update(self.pred_pair_confidence(
                                torch.Tensor(side_score[np.ix_(gid1.tolist(), (gid2 - side_score.shape[0]).tolist())])
                                * self.side_weight))
                        # 所有模态求和，生成pair
                        # scores += torch.Tensor(side_score[np.ix_(rows_to_keep,cols_to_keep)]) * self.side_weight
                        # new_pair.update(self.pred_pair_confidence(scores / float(i + 2)))

                        # 每个模态都单独生成pair
                        # s = torch.Tensor(side_score[np.ix_(rows_to_keep,cols_to_keep)] * self.side_weight)
                        # new_pair.update(self.pred_pair_confidence(s))

                        # scores += torch.Tensor(side_score[rows_to_keep, cols_to_keep]) * self.side_weight
                        # new_pair.update(self.pred_pair_confidence(scores / float(i + 2)))

                    print('new_pair len: {}'.format(len(new_pair)))
                    # 方案1：删除conflict的pair
                    if self.csp == 1:
                        new_pair = self.delete_repeat_pair(new_pair)
                    elif self.csp == 2:
                        new_pair = self.choose_repeat_pair(new_pair)
                    else:
                        pass

                    new_pair_ = torch.tensor(list(new_pair)).to(self.device)

                    # self.train_pair = torch.cat((self.train_pair, new_pair_), dim=0)
                    self.train_pair_confidence = torch.cat((self.train_pair_confidence, new_pair_), dim=0)


                    del new_pair_
                    torch.cuda.empty_cache()

                    print("\n*********************************************")
                    # print(self.train_pair.shape)
                    print(self.train_pair_confidence.shape)
                    print("increase pseudo pair: {}".format(len(new_pair)))
                    print("*********************************************\n")
                    trainset.append(round(len(new_pair), 4))

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

        # save_true_excel()

    def pred_pair(self, score):
        new_set = set()
        A = score.argmax(axis=0)
        B = score.argmax(axis=1)
        for i, j in enumerate(A):
            if B[j] == i:
                new_set.add((self.rest_set_1[j], self.rest_set_2[i]))

        return new_set

    def pred_pair_s(self, score):
        new_set = set()
        A = score.argmax(axis=0)
        B = score.argmax(axis=1)
        for i, j in enumerate(A):
            if B[j] == i and score[j][i] > self.thred:
                new_set.add((self.rest_set_1[j], self.rest_set_2[i]))

        return new_set

    def pred_pair_confidence(self, score):

        new_set = set()
        A = score.argmax(axis=0)
        B = score.argmax(axis=1)
        for i, j in enumerate(A):
            if B[j] == i and (score[j][i] > self.sig or score[i][j] > self.sig):
                if score[j][i] < self.sig or score[i][j] < self.sig:
                    sc = (score[j][i] + score[i][j]) / 2
                    new_conf = math.exp(-0.5 * (sc - self.sig) ** 2)
                else:
                    new_conf = 1
                if self.confidence:
                    new_set.add((self.rest_set_1[j], self.rest_set_2[i], new_conf))
                else:
                    # 当score全部为1：代表没有confidence weight
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
                # 存在相同的a对应不同的b
                for b in bs:
                    conflicting_tuples.add((a, b))
        for b, a_s in b_to_as.items():
            if len(a_s) > 1:
                # 存在相同的a对应不同的b
                for a in a_s:
                    conflicting_tuples.add((a, b))
        # conflicting_tuples {(1141, 24916), (10841, 24601), (10841, 16563), (1141, 22302)}
        print("conflicting_tuples: {}".format(len(conflicting_tuples)))

        conflicting_tuples_v = set()
        for (x, y) in conflicting_tuples:
            for triple in pair:
                if triple[0] == x and triple[1] == y:
                    # print(triple)
                    conflicting_tuples_v.add((triple[0], triple[1], triple[2]))
                    # print(conflicting_tuples_v)

        new_pair = pair.difference(conflicting_tuples_v)
        # new_pair = pair.difference(conflicting_tuples)
        return new_pair

    def choose_repeat_pair(self, pair):
        max_triplets = {}
        for triplet in pair:
            x, y, z = triplet
            # 更新字典以 x 为键
            if x not in max_triplets or max_triplets[x][2] < z:
                max_triplets[x] = triplet
            # 更新字典以 y 为键
            if y not in max_triplets or max_triplets[y][2] < z:
                max_triplets[y] = triplet

        new_pair = set()
        for triplet in max_triplets.values():
            x, y, z = triplet
            # 确保三元组是当前最大值
            if (x not in max_triplets or max_triplets[x] == triplet) and (
                    y not in max_triplets or max_triplets[y] == triplet):
                new_pair.add(triplet)

        return new_pair

    def sim_results(self, Matrix_A, Matrix_B):
        # A x B.t
        A_sim = torch.mm(Matrix_A, Matrix_B.t())
        return A_sim

    def test_rank(self, sourceVec, targetVec):
        sim_mat = self.sim_results(sourceVec, targetVec)
        mr = 0
        mrr = 0
        top_k = [1, 3, 10]
        hits = [0] * len(top_k)
        hits1_rest = set()
        pair = set()
        pred_pair = set()
        sim_list = []
        sim_2 = []
        for i in range(sourceVec.shape[0]):
            # print(i)
            gold = i
            rank = (-sim_mat[i, :]).argsort()
            # print(rank)
            # print(entity)
            # rank = entity[rank]

            pair.add((i, rank[0]))
            hits1_rest.add((gold, rank[0]))

            # 超过置信度0.9的样本归到训练集中
            if sim_mat[i, rank[0]] > 0.9:
                pred_pair.add((i, rank[0]))
            # if i == 5:
            #     exit(0)
            sim_list.append((i, gold, rank[0], sim_mat[i, gold], sim_mat[i, rank[0]]))
            # print(i, gold, rank[0],sim_mat[i, gold], sim_mat[i, rank[0]])
            assert gold in rank
            rank_index = np.where(rank == gold)[0][0]
            # print(rank_index)
            mr += (rank_index + 1)
            mrr += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    hits[j] += 1
            # print(hits[0])

        mrr /= targetVec.shape[0]
        print("-------------------------------------------")
        print('hits1 is ', hits[0] / sourceVec.shape[0])
        print('hits3 is ', hits[1] / sourceVec.shape[0])
        print('hits10 is ', hits[2] / sourceVec.shape[0])
        print('mrr is ', mrr)

        # print('average of sim_mat[i, rank[0]] is :' + str(sum(sim_2)/len(sim_2)))
        print("-------------------------------------------")
        return hits[0] / sourceVec.shape[0], pair, sim_list, pred_pair

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

    # def sinkhorn_test(self, sourceVec, targetVec, device, len_dev):
    def sinkhorn_test(self, sourceVec, targetVec, device, len_dev, gid1, gid2):

        st_sim_mat = self.sim_results(sourceVec, targetVec)
        st_sim_mat = st_sim_mat * self.ST_weight
        sim_mat_list = []
        sim_mat_list.append(st_sim_mat.detach().numpy())
        if self.exp_mod:
            for name, side_score in self.side_modalities.items():
                if 'Att' in name or 'Name' in name:
                    sim_mat_list.append(side_score)
                    # sim_mat_list.append(side_score[np.ix_(gid1.tolist(), (gid2 - side_score.shape[0]).tolist())])
                else:
                    sim_mat_list.append(side_score[np.ix_(gid1.tolist(), (gid2-side_score.shape[0]).tolist())])

        try:
            weight_norm = F.softmax(self.weight_t, dim=0)
            weight_norm_ = weight_norm.detach().cpu().numpy()
            fuse_data = [weight_norm_[idx] * sim_mat_list[idx] for idx in range(len(sim_mat_list))]
            sim_mat = sum(fuse_data) / len(sim_mat_list)

        except:
            sim_mat = sum(sim_mat_list) / len(sim_mat_list)

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
        if sim_mat_r.ndim == 3:
            M = sim_mat_r
        else:
            try:
                M = sim_mat_r.view(1, sim_mat_r.size(0), -1)
            except:
                M = sim_mat_r.reshape(1, sim_mat_r.shape[0], -1)
                M = torch.tensor(M, dtype=torch.float32)
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

    def sinkhorn_test_2(self, st_sourceVec, st_targetVec, se_sourceVec, se_targetVec, device, len_dev):
        st_sim_mat = self.sim_results(st_sourceVec, st_targetVec)
        se_sim_mat = self.sim_results(se_sourceVec, se_targetVec)
        sim_mat = st_sim_mat + se_sim_mat
        count = 0
        if self.exp_mod:
            for _, side_score in self.side_modalities.items():
                sim_mat += torch.Tensor(side_score)
                count = count + 1
        # sim_mat = sim_mat / float(count+1)
        print(count)
        sim_mat = sim_mat / float(count + 2)

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
        m, n = sim_mat_r.shape
        a = torch.ones([1, m], requires_grad=False, device=device)
        b = torch.ones([1, n], requires_grad=False, device=device)
        P = sinkhorn(a, b, M, 0.02, max_iters=100, stop_thresh=1e-3)
        P = view2(P)

        # evaluate_sim
        result = evaluate_sim_matrix(link=torch.stack([torch.arange(len_dev),
                                                       torch.arange(len_dev)], dim=0),
                                     sim_x2y=P,
                                     no_csls=True)
        return result, st_sim_mat, se_sim_mat


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



