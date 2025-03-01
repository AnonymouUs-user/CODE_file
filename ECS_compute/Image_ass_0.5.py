import argparse
import numpy as np
from tqdm import tqdm
import torch
import pickle
from utils import *

def sinkhorn_test(scores, len_pair,device):
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


parser = argparse.ArgumentParser(description='side_ass process (Vis) for PathFusion')
parser.add_argument('--dataset', type=str, default='DB15K-FB15K', help='dataset name', choices=['DB15K-FB15K', 'YAGO15K-FB15K'])
parser.add_argument('--max_image_num', type=int, default=6, help='max image num for each entity', choices=[1, 2, 3, 4, 5, 6])

args = parser.parse_args()


# dataset_path_list = ['FB15K-YAGO15K' ]
dataset_path_list = ['YAGO15K-FB15K']
for dataset_path in dataset_path_list:
    print(dataset_path)
    source_dataset, target_dataset = dataset_path.split('-')
    dataset = target_dataset + '-' + source_dataset
    source_dataset = source_dataset.lower()
    target_dataset = target_dataset.lower()
    print(source_dataset)
    print(target_dataset)
    use_img_num = args.max_image_num
    print('use_img_num: {}'.format(use_img_num))

    if source_dataset == 'db15k':
        from utils_db_0_5 import *
    else:
        from utils_yb_0_5 import *

    # load source image embedding
    with open('../data/image_embed_1/{}.npy'.format(source_dataset), 'rb') as f:
        source_embedding = pickle.load(f)
        # source_embedding = np.load(f)

    # load source entity id to image id mapping
    with open('../data/image_embed_1/{}'.format(source_dataset), 'rb') as f:
        source_id2img = pickle.load(f)

    print('source embedding shape: {}'.format(source_embedding.shape))
    print('source id2img length: {}'.format(len(source_id2img)))

    # load target image embedding
    with open('../data/image_embed_1/{}.npy'.format(target_dataset), 'rb') as f:
        target_embedding = pickle.load(f)

    # load target entity id to image id mapping
    with open('../data/image_embed_1/{}'.format(target_dataset), 'rb') as f:
        target_id2img = pickle.load(f)

    print('target embedding shape: {}'.format(target_embedding.shape))
    print('target id2img length: {}'.format(len(target_id2img)))

    print('source target entity num:', len(source2id), len(target2id))
    image_scores = np.zeros((len(source2id), len(target2id)))
    image_scores = -float('inf') * np.ones((len(source2id), len(target2id)))

    # scores = np.zeros((len(source2id), len(target2id)), dtype=np.float32)
    print('source_id vs target_id')

    print(len(source2id))
    print(len(target2id))

    for i in tqdm(range(len(source2id))):
        for j in range(len(target2id)):
            for ii in range(min(use_img_num, len(source_id2img[i]))):
                for jj in range(min(use_img_num, len(target_id2img[j]))):
                    image_scores[i, j] = max(image_scores[i, j],
                                             np.dot(source_embedding[source_id2img[i][ii]],
                                                    target_embedding[target_id2img[j][jj]]))


    dev_pair = np.array(dev_pair)
    scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
    for i, l in enumerate(dev_pair[:, 0]):
        for j, r in enumerate(dev_pair[:, 1]):
            scores[i][j] = image_scores[l][r - len(source2id)] if image_scores[l][r - len(source2id)] != -float('inf') else 0

    print(scores.shape)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    print(scores.shape)

    np.save(f'../data/ECS_results/seed0.5/{dataset}/Vis_1.npy', scores)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    sinkhorn_test(scores, scores.shape[0], device=device)

