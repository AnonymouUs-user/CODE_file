import numpy as np
import torch
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

vis_path = '/home/wluyao/EIEA/data/ECS_results/seed0.8/FB15K-DB15K/Vis.npy'
moda_np = np.load(vis_path)
m, n = moda_np.shape
moda_np_1 = moda_np[:m//2, :n//2]
print(moda_np_1.shape)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
sinkhorn_test(moda_np_1, moda_np_1.shape[0], device=device)