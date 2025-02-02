from pkl_read import *
import numpy as np
from tqdm import tqdm
import torch

dataset = ['ja_en', 'fr_en', 'zh_en']
for ds in dataset:
    pair, len_pair = pair_load(ds)
    pair = np.array(pair)
    ent_1, ent_2 = ent_load(ds)
    ill_pair = ill_pair_load(ds)
    image_source_dir = pkl_load(ds)
    # print(len_pair) pair length:15000

    ent1_id_list = list(ent_1.keys())
    ent2_id_list = list(ent_2.keys())

    print('ent_1_min:', min(ent1_id_list))
    print('ent_1_max:', max(ent1_id_list))
    print('ent_2_min:', min(ent2_id_list))
    print('ent_2_max:', max(ent2_id_list))

    r_index_1, r_index_2 = load_reverse_index(ds)
    len_ent_1 = len(ent_1)
    len_ent_2 = len(ent_2)

    image_scores = -float('inf') * np.ones((len_pair, len_pair))

    for i in tqdm(range(len(ent_1))):
        for j in range(len(ent_2)):
            try:
                image_scores[i, j] = max(image_scores[i, j],
                                         np.dot(image_source_dir[r_index_1[str(ent1_id_list[i])]],
                                                image_source_dir[r_index_2[str(ent2_id_list[j])]]))
                # print(image_scores[i, j])
            except:
                continue
                # print([i, j])

    ill_pair = np.array(ill_pair)
    scores = np.zeros((len(ill_pair), len(ill_pair)), dtype=np.float32)
    print(scores.shape)
    for i, l in enumerate(ill_pair[:, 0]):
        for j, r in enumerate(ill_pair[:, 1]):
            scores[i][j] = image_scores[l][r - len_ent_1] if image_scores[l][r - len_ent_1] != -float('inf') else 0

    print(scores.shape)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    file_name = '../data/DBP15K/DBP_ECS_1/new_Vis_{}.npy'.format(ds)
    np.save(file_name, scores)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    sinkhorn_test(scores, scores.shape[0], device=device)

