import torch
from utils import *
import re
from os.path import join as pjoin
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm, trange
from gme_inference import GmeQwen2VL

def date2float(date):
    if re.match(r'\d+-\d+-\d+', date):
        year = date.split('-')[0]
        mouth = date.split('-')[1]
        decimal_right = '0' if mouth == '12' else str(int(mouth) / 12)[2:]
        if mouth == '12':
            year = str(int(year) + 1)
        return year + '.' + decimal_right
    else:
        return date

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

def load_attr(dt):
    if 'DB' in dt:
        attr_path = '../data/MMEA-data/attr/FB_DB_attr'
        ent_path = '../data/ent/FB-DB'
    else:
        attr_path = '../data/MMEA-data/attr/FB_YAGO_attr'
        ent_path = '../data/ent/FB-YAGO'

    source_keys = set()
    target_keys = set()
    source2id = {}
    target2id = {}
    id2source = {}
    id2target = {}
    ent2id = {}
    id2ent = {}
    id2attrs = []

    # load_entity
    with open(pjoin(ent_path, 'ent_ids_1'), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_name = line.split('\t')
            ent2id[ent_name] = int(ent_id)
            id2ent[int(ent_id)] = ent_name
            source2id[ent_name] = int(ent_id)
            id2source[int(ent_id)] = ent_name
    with open(pjoin(ent_path, 'ent_ids_2'), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_name = line.split('\t')
            ent2id[ent_name] = int(ent_id)
            id2ent[int(ent_id)] = ent_name
            target2id[ent_name] = int(ent_id) - len(source2id)  # 从0开始计数
            id2target[int(ent_id) - len(source2id)] = ent_name

    # load_attr
    s_en_num = len(source2id)
    t_en_num = len(target2id)
    s_en_onehot = np.ones((s_en_num, 1))
    t_en_onehot = np.ones((t_en_num, 1))
    with open(attr_path, 'r') as f:
        try:
            for i, line in enumerate(f):
                line = line.strip()
                tmp = []
                if i < len(source2id):
                    attr_value_schema = line.split('^^^')
                    for attr_value_schema in attr_value_schema:
                        if attr_value_schema == '':
                            s_en_onehot[i][0]=0
                            continue
                        attr, value, schema = attr_value_schema.split('|||')  # KG1 schema
                        tmp.append((attr, value, schema))
                    id2attrs.append(tmp)
                else:
                    attr_values = line.split('^^^')
                    for attr_value in attr_values:
                        if attr_value == '':
                            t_en_onehot[i - len(source2id)][0] = 0
                            continue
                        attr, value = attr_value.split('|||')  # KG2 schema
                        tmp.append((attr, value))
                    id2attrs.append(tmp)
        except:
            print('Attr error!')
    print(s_en_onehot.sum())

    source_attr_value_set = set()
    target_attr_value_set = set()
    for i, attrs in enumerate(id2attrs):
        if i < len(source2id):
            for attr, value, schema in attrs:
                value = date2float(value)
                source_attr_value_set.add(attr+' '+value)
                source_keys.add(attr)
        else:
            for attr, value in attrs:
                value = date2float(value)
                target_attr_value_set.add(attr + ' ' + value)
                target_keys.add(attr)
    source_attr_value_list, \
    target_attr_value_list = sorted(list(source_attr_value_set)),\
                                                     sorted(list(target_attr_value_set))

    source2attr = np.zeros((len(source2id), len(source_attr_value_set)), dtype=np.float32)
    target2attr = np.zeros((len(target2id), len(target_attr_value_set)), dtype=np.float32)
    for i, attrs in enumerate(id2attrs):
        if i < len(source2id):
            for attr, value, schema in attrs:
                value = date2float(value)
                pos = source_attr_value_list.index(attr + ' ' + value)
                source2attr[i][pos] = 1
        else:
            for attr, value in attrs:
                value = date2float(value)
                pos = target_attr_value_list.index(attr + ' ' + value)
                target2attr[i - len(source2id)][pos] = 1
    return source_attr_value_list, target_attr_value_list, source2attr, target2attr

def attr_emb(s_attr_list, t_attr_list, device, ):
    model = GmeQwen2VL("../pre_train_model/gme-Qwen2-VL-2B-Instruct")

    # model = SentenceTransformer(
    #     '../pre_train_model/Roberta_finetuning_semantic_similarity_stsb_multi_mt/').to(device)
    # model.eval()
    source_key_embeddings = []
    target_key_embeddings = []
    source_value = []
    target_value = []
    batch_size = 128
    for i in trange(0, len(s_attr_list), batch_size):
        key_sents = s_attr_list[i:i + batch_size]

        for j in range(len(key_sents)):
            try:
                source_value.append(float(key_sents[j].split(' ')[1]))
            except:
                source_value.append(0)
            key_sents[j] = key_sents[j].split(' ')[0]

        # source_key_embeddings.append(model.encode(key_sents))
        source_key_embeddings.append(model.get_text_embeddings(texts=key_sents))


    source_key_embeddings = np.concatenate(source_key_embeddings, axis=0)
    print('********************source_over*****************************************************')

    for i in tqdm(range(0, len(t_attr_list), batch_size)):
        target_key_sents = t_attr_list[i:i + batch_size]
        for j in range(len(target_key_sents)):
            try:
                target_value.append(float(target_key_sents[j].split(' ')[1]))
            except:
                target_value.append(0)
            target_key_sents[j] = target_key_sents[j].split(' ')[0]
        # target_key_embeddings.append(model.encode(target_key_sents))
        target_key_embeddings.append(model.get_text_embeddings(texts=target_key_sents))

    target_key_embeddings = np.concatenate(target_key_embeddings, axis=0)
    print('********************target_over*****************************************************')

    source_value = np.array(source_value)[:, np.newaxis]  # np.newaxis 插入新维度 source_value.shape (25796, 1)
    target_value = np.array(target_value)[np.newaxis, :]  # target_value.shape (1, 17134)

    # scores_key = np.matmul(source_key_embeddings, target_key_embeddings.T)

    import torch
    source_name_embedding_gpu = torch.tensor(source_key_embeddings, device='cuda')
    target_name_embedding_gpu = torch.tensor(target_key_embeddings, device='cuda')
    scores_all_gpu = torch.matmul(source_name_embedding_gpu, target_name_embedding_gpu.T)
    scores_key = scores_all_gpu.cpu().numpy()


    # scores_key.shape (25796, 17134)
    scores_value = 1 / (np.abs(source_value - target_value) + 1e-3)
    attr2attr = scores_key * scores_value
    return attr2attr

def score_sample(pairs, score):
    s_len, t_len = score.shape
    pair_len = len(pairs)
    pair_score = np.zeros((pair_len, pair_len), dtype=np.float32)
    for i in range(pair_len):
        for j in range(pair_len):
            pair_score[i][j] = score[pairs[i][0]][pairs[j][1] - s_len]
    print("pair_score shape:", pair_score.shape)
    return pair_score

def pair_load(dt):
    pairs=[]
    with open(pjoin('../data/MMEA-data/seed/', dt, 'ref_ent_ids'), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id1, ent_id2 = line.split('\t')
            pairs.append((int(ent_id1), int(ent_id2)))
    return pairs

if __name__ =='__main__':
    dataset_set = ['FB15K_YAGO15K','FB15K_DB15K']
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    for dt in dataset_set:
        print(dt)
        s_attr_list, t_attr_list, s2attr, t2attr = load_attr(dt)
        attr2attr = attr_emb(s_attr_list, t_attr_list, device)
        score = s2attr @ attr2attr @ t2attr.T
        score = (score-score.min())/(score.max()-score.min())
        pair_score = score_sample(pair_load(dt), score)
        np.save(f'../data/ES_ALL/{dt}_Qwen2b_attr.npy', score)
        test_sinkhorn(score, score.shape[0], device=device)
