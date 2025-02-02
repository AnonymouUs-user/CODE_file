import torch
from utils import *
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm, trange
from os.path import join as pjoin
from gme_inference import GmeQwen2VL

# 加载实体ent集
def ent_load(dataset_name):

    pass

# 加载对应的Name/Name_emb():
def ent_Name_load(dataset_name, device):
    # s_name, t_name = dataset_name.split()
    if 'DB' in dataset_name:
        source_name_path = '../data/MMEA_name/DB_name.txt'
        target_name_path = '../data/MMEA_name/FB_DB_name.txt'
    else:
        source_name_path = '../data/MMEA_name/YAGO_name.txt'
        target_name_path = '../data/MMEA_name/FB_YAGO_name.txt'
    sourceid_name = {}
    targetid_name = {}
    with open(source_name_path, 'r') as f:
        for line in f:
            line = line.strip()
            id, name = line.split(' ')
            sourceid_name[int(id)] = name
    with open(target_name_path, 'r') as f:
        for line in f:
            line = line.strip()
            id, name = line.split(' ')
            targetid_name[int(id)] = name

    model = GmeQwen2VL("../pre_train_model/gme-Qwen2-VL-2B-Instruct")

    # model = SentenceTransformer(
    #     '../pre_train_model/Roberta_finetuning_semantic_similarity_stsb_multi_mt/').to(device)
    # model.eval()
    batch_size = 128
    source_name_embedding = []
    target_name_embedding = []
    sourceid_name_lt = list(sourceid_name.values())
    targetid_name_lt = list(targetid_name.values())

    for i in trange(0, len(sourceid_name_lt), batch_size):
        key_sents = sourceid_name_lt[i:i + batch_size]
        # source_name_embedding.append(model.encode(key_sents))
        source_name_embedding.append(model.get_text_embeddings(texts=key_sents))
    source_name_embedding = np.concatenate(source_name_embedding, axis=0)

    for i in trange(0, len(targetid_name_lt), batch_size):
        key_sents = targetid_name_lt[i:i + batch_size]
        # target_name_embedding.append(model.encode(key_sents))
        target_name_embedding.append(model.get_text_embeddings(texts=key_sents))

    target_name_embedding = np.concatenate(target_name_embedding, axis=0)

    # scores_all = np.matmul(source_name_embedding, target_name_embedding.T)

    import torch
    source_name_embedding_gpu = torch.tensor(source_name_embedding, device='cuda')
    target_name_embedding_gpu = torch.tensor(target_name_embedding, device='cuda')
    scores_all_gpu = torch.matmul(source_name_embedding_gpu, target_name_embedding_gpu.T)
    scores_all = scores_all_gpu.cpu().numpy()

    print("score_all shape:", scores_all.shape)
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
    pair_score = np.zeros((pair_len, t_len), dtype=np.float32)
    for i in range(pair_len):
        for j in range(pair_len):
            pair_score[i][j] = score[pairs[i][0]][pairs[j][1] - s_len]
    print("pair_score shape:", pair_score.shape)
    return pair_score


def test_sinkhorn(scores, len_pair,device):
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
    dataset_set = ['FB15K_DB15K', 'FB15K_YAGO15K']
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    for dt in dataset_set:
        print(dt)
        score, shape_0 = ent_Name_load(dt, device)
        pair_score = score_sample(pair_load(dt), score)
        # 保存的是原实体集顺序对应的score，因为train集会打乱，不按照pair_score
        # 的顺序
        np.save(f'../data/ES_ALL/{dt}_Qwen2b_name.npy', score)
        test_sinkhorn(pair_score, pair_score.shape[0], device)

