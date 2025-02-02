import torch
from utils import *
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm, trange
from os.path import join as pjoin
import pickle

# 加载实体ent集
def ent_load(dt):
    ent1_dir={}
    ent2_dir = {}
    if 'DB' in dt:
        ent1_path = '../data/ent/FB-DB/ent_ids_1'
        ent2_path = '../data/ent/FB-DB/ent_ids_2'
    else:
        ent1_path = '../data/ent/FB-YAGO/ent_ids_1'
        ent2_path = '../data/ent/FB-YAGO/ent_ids_2'
    with open(ent1_path, 'r') as f:
        for line in f:
            id1, ent = line.split('\t')
            ent1_dir[int(id1)] = ent
    with open(ent2_path, 'r') as f:
        for line in f:
            id1, ent = line.split('\t')
            ent2_dir[int(id1)] = ent
    return ent1_dir, ent2_dir

# 加载对应的Image/Image_emb():
def ent_Image_load(dataset_name, s_ent_len, t_ent_len):
    use_img_num = 6
    if 'DB' in dataset_name:
        source_image_path = '../data//image_embed_1/db15k.npy'
        source_id2img_path = '../data//image_embed_1/db15k'
        target_image_path = '../data//image_embed_1/fb15k.npy'
        target_id2img_path = '../data//image_embed_1/fb15k'
    else:
        source_image_path = '../data//image_embed_1/yago15k.npy'
        source_id2img_path = '../data//image_embed_1/yago15k'
        target_image_path = '../data//image_embed_1/fb15k.npy'
        target_id2img_path = '../data//image_embed_1/fb15k'

    with open(source_image_path, 'rb') as f:
        source_image_emb = pickle.load(f)
    with open(source_id2img_path, 'rb') as f:
        source_id2img = pickle.load(f)
    with open(target_image_path, 'rb') as f:
        target_image_emb = pickle.load(f)
    with open(target_id2img_path, 'rb') as f:
        target_id2img = pickle.load(f)

    # scores = np.zeros((len(source_id2img), len(target_id2img)))
    scores = -float('inf') * np.ones((s_ent_len, t_ent_len))
    print(source_image_emb.shape)
    scores_all = np.zeros((s_ent_len, t_ent_len), dtype=np.float32)

    for i in tqdm(range(s_ent_len)):
        for j in range(t_ent_len):
            for ii in range(min(use_img_num, len(source_id2img[i]))):
                for jj in range(min(use_img_num, len(target_id2img[j]))):
                    scores[i, j] = max(scores[i, j], np.dot(source_image_emb[source_id2img[i][ii]],
                                                    target_image_emb[target_id2img[j][jj]]))
                    if scores[i, j] == -float('inf'):
                        scores_all[i, j] = 0
                    else:
                        scores_all[i, j] = scores[i, j]

    scores_all = (scores_all - np.min(scores_all)) / (np.max(scores_all) - np.min(scores_all))
    return scores_all, scores_all.shape[0]

def pair_load(dt):
    pairs=[]
    with open(pjoin('../data/MMEA-data/seed/', dt, 'ref_ent_ids'), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id1, ent_id2 = line.split('\t')
            pairs.append((int(ent_id1), int(ent_id2)))
    return pairs

def score_sample(pairs, score):
    s_len, t_len = score.shape
    pair_len = len(pairs)
    pair_score =  np.zeros((pair_len, t_len), dtype=np.float32)
    for i in range(pair_len):
        for j in range(pair_len):
            pair_score[i][j] = score[pairs[i][0]][pairs[j][1] - s_len]
    print("pair_score shape:", pair_score.shape)
    return pair_score


def test_sinkhorn(scores, len_pair, device):
    scores = torch.Tensor(scores).to(device)
    sim_mat_r = 1 - scores
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
    result = evaluate_sim_matrix(link=torch.stack([torch.arange(len_pair),
                                                   torch.arange(len_pair)], dim=0),
                                 sim_x2y=P,
                                 no_csls=True)
    return result

# main函数
if __name__ =='__main__':
    dataset_set = ['FB15K_YAGO15K', 'FB15K_DB15K']
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    for dt in dataset_set:
        print(dt)
        ent1_dir, ent2_dir = ent_load(dt)
        score, shape_0 = ent_Image_load(dt, len(ent1_dir), len(ent2_dir))
        pair_score = score_sample(pair_load(dt), score)
        # 保存的是原实体集顺序对应的score，因为train集会打乱，不按照pair_score
        # 的顺序
        np.save(f'../data/ES_ALL/{dt}_Image_2.npy', score)
        test_sinkhorn(pair_score, pair_score.shape[0], device)

