import numpy as np
from pkl_read import ent_load, ill_pair_load, sinkhorn_test
from tqdm import tqdm, trange
from sentence_transformers import SentenceTransformer
import torch

def att_load(dataset):
    att_1_keys = set()
    id1_attr = []
    with open('../data/attr_trans/{}_attr_1.txt'.format(dataset), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            attr_schemas = line.strip().split(' ')
            attr_l_1 = []
            for attr in attr_schemas:
                att_1_keys.add(attr)
                attr_l_1.append(attr)
            id1_attr.append(attr_l_1)

    att_2_keys = set()
    id2_attr = []
    with open('../data/attr_trans/{}_attr_2.txt'.format(dataset), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            attr_schemas = line.strip().split('||')
            attr_l_2 = []
            for attr in attr_schemas:
                att_2_keys.add(attr)
                attr_l_2.append(attr)
            id2_attr.append(attr_l_2)

    print('{} attribute num:{}'.format(dataset, len(att_1_keys)))
    print('{} attribute num:{}'.format(dataset, len(att_2_keys)))
    return att_1_keys, att_2_keys, len(att_1_keys), len(att_2_keys), id1_attr, id2_attr

def att_load_1(dataset):
    ent1_att_path = '../data/DBP15K/{}/training_attrs_1'.format(dataset)
    ent2_att_path = '../data/DBP15K/{}/training_attrs_2'.format(dataset)
    att_1 = {}
    att_2 = {}

    with open(ent1_att_path, 'r') as f:
        for line in f:
            line = line.strip()
            l = line.split('\t')
            new_l = [item for item in l[1:]]
            att_1[l[0]] = new_l

    with open(ent2_att_path, 'r') as f:
        for line in f:
            line = line.strip()
            l = line.split('\t')
            new_l = [item for item in l[1:]]
            att_2[l[0]] = new_l
    return att_1, att_2


def att_load_notranslate(ent_1, ent_2, att_1, att_2):
    id_att_1 = {}
    id_att_2 = {}
    att_1_keys = set()
    att_2_keys = set()
    id1_attr = []
    id2_attr = []

    for id in ent_1.keys():
        if ent_1[id] in att_1.keys():
            att_l = att_1[ent_1[id]]
            id_att_1[int(id)] = att_l
        else:
            id_att_1[int(id)] = 'No_property'

    for id in ent_2.keys():
        if ent_2[id] in att_2.keys():
            att_l = att_2[ent_2[id]]
            id_att_2[int(id)] = att_l
        else:
            id_att_2[int(id)] = 'No_property'

    for att in id_att_1.values():
        id1_attr.append(att)
        for a in att:
            att_1_keys.add(a)
    for att in id_att_2.values():
        id2_attr.append(att)
        for a in att:
            att_2_keys.add(a)

    print('{} attribute num:{}'.format(d, len(att_1_keys)))
    print('{} attribute num:{}'.format(d, len(att_2_keys)))
    print(id1_attr[0])
    print(id2_attr[0])

    return att_1_keys, att_2_keys, len(att_1_keys), len(att_2_keys), id1_attr, id2_attr

def translate_att():
    """
    翻译att，并且保持id的顺序,保存翻译后的文档
    """
    from transformers import MBartForConditionalGeneration, MBart50Tokenizer
    dataset = ['zh_en']
    trans_model = MBartForConditionalGeneration.from_pretrained('../mbart_large_50_many_to_many_mmt')
    trans_tokenizer = MBart50Tokenizer.from_pretrained('../mbart_large_50_many_to_many_mmt')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trans_model.to(device)
    trans_model.eval()

    def trans_list(sentences, src_lan):
        if src_lan == 'zh':
            src_lang = 'zh_CN'
        elif src_lan == 'ja':
            src_lang = 'ja_XX'
        else:
            src_lang = 'fr_XX'

        target_lang = "en_XX"
        translated_texts = []
        trans_tokenizer.src_lang = src_lang
        batch_size = 32
        # for i in tqdm(range(0, len(sentences), batch_size), desc="Translating", unit="batch"):
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            encoded_input = trans_tokenizer(batch_sentences,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True,
                                            max_length=512).to(device)
            generated_tokens = trans_model.generate(**encoded_input,
                                                    forced_bos_token_id=trans_tokenizer.lang_code_to_id[target_lang])
            result = trans_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            translated_texts.extend(result)
        return translated_texts

    for d in dataset:
        print(d)
        ent_1, ent_2 = ent_load(d)
        att_1, att_2 = att_load_1(d)
        src_lang, tgt_lang = d.split('_')
        trans_id_att_1 = {}
        id_att_2 = {}
        # for id_1 in ent_1.keys():
        for id_1 in tqdm(ent_1.keys()):
            if ent_1[id_1] in att_1.keys():
                att_list = att_1[ent_1[id_1]]
                att_tran_list = trans_list(att_list, src_lang)
                trans_id_att_1[id_1] = att_tran_list
            else:
                trans_id_att_1[id_1] = ['NO_property']
        # for id_2 in ent_2.keys():
        for id_2 in tqdm(ent_2.keys()):
            if ent_2[id_2] in att_2.keys():
                att_list = att_2[ent_2[id_2]]
                id_att_2[id_2] = att_list
            else:
                id_att_2[id_2] = ['NO_property']

        # save dictionary trans_id_att_1
        with open(f'../data/attr_trans/{d}_id_att_1', 'w') as f:
            for k,v in trans_id_att_1.items():
                f.write(f'{k}\t{" ".join(v)}\n')
        # save dictionary id_att_2
        with open(f'../data/attr_trans/{d}_id_att_2', 'w') as f:
            for k,v in id_att_2.items():
                f.write(f'{k}\t{" ".join(v)}\n')

# translate_att()
# exit(0)
def att_load_translate(dataset):
    """
    加载翻译后的属性
    """
    id_att_1 = {}
    id_att_2 = {}
    with open(f'../data/attr_trans/{dataset}_id_att_1', 'r') as f:
        for line in f:
            id_1, att_list = line.strip().split('\t')
            id_att_1[id_1] = att_list.split(' ')
    with open(f'../data/attr_trans/{dataset}_id_att_2', 'r') as f:
        for line in f:
            id_2, att_list = line.strip().split('\t')
            id_att_2[id_2] = att_list.split(' ')

    att_1_keys = set()
    att_2_keys = set()
    id1_attr = []
    id2_attr = []

    for att in id_att_1.values():
        id1_attr.append(att)
        for a in att:
            att_1_keys.add(a)
    for att in id_att_2.values():
        id2_attr.append(att)
        for a in att:
            att_2_keys.add(a)

    print('{} attribute num:{}'.format(d, len(att_1_keys)))
    print('{} attribute num:{}'.format(d, len(att_2_keys)))
    # print(id1_attr[0])
    # print(id2_attr[0])

    return att_1_keys, att_2_keys, len(att_1_keys), len(att_2_keys), id1_attr, id2_attr

# dataset = ['ja_en', 'fr_en', 'zh_en']
dataset = ['zh_en']
for d in dataset:
    print(d)
    # att_1_keys, att_2_keys, len_att_1, len_att_2, id1_attr, id2_attr = att_load(d)
    # ent_1, ent_2 = ent_load(d)
    # att_1, att_2 = att_load_1(d)
    # att_1_keys, att_2_keys, len_att_1, len_att_2, id1_attr, id2_attr = att_load_notranslate(ent_1, ent_2, att_1, att_2)
    att_1_keys, att_2_keys, len_att_1, len_att_2, id1_attr, id2_attr = att_load_translate(d)
    # ent_1, ent_2 = ent_load(d)
    source2attr = np.zeros((len(id1_attr), len_att_1), dtype=np.float32)
    target2attr = np.zeros((len(id2_attr), len_att_2), dtype=np.float32)
    source_attr_list = sorted(list(att_1_keys))
    target_attr_list = sorted(list(att_2_keys))
    for i, attrs, in enumerate(id1_attr):
        for attr in attrs:
            pos = source_attr_list.index(attr)
            source2attr[i, pos] = 1
    for i, attrs, in enumerate(id2_attr):
        for attr in attrs:
            pos = target_attr_list.index(attr)
            target2attr[i, pos] = 1

    print(source2attr.shape)
    print(target2attr.shape)

    source_key_embeddings = []
    target_key_embeddings = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(
        '../Roberta_finetuning_semantic_similarity_stsb_multi_mt/').to(device)
    # model = SentenceTransformer('../LaBSE/').to(device)
    model.eval()

    batch_size = 128
    for i in trange(0, len(source_attr_list), batch_size):
        key_sents = source_attr_list[i:i+batch_size]
        source_key_embeddings.append(model.encode(key_sents, convert_to_tensor=True, show_progress_bar=False).to(device))
    source_key_embeddings = torch.cat(source_key_embeddings, dim=0)
    source_key_embeddings = source_key_embeddings.detach().cpu().numpy()

    for i in trange(0, len(target_attr_list), batch_size):
        key_sents = target_attr_list[i:i+batch_size]
        target_key_embeddings.append(model.encode(key_sents, convert_to_tensor=True, show_progress_bar=False).to(device))
    target_key_embeddings = torch.cat(target_key_embeddings, dim=0)
    target_key_embeddings = target_key_embeddings.detach().cpu().numpy()

    att2att = np.matmul(source_key_embeddings, target_key_embeddings.T)
    print('att2att.shape: ', att2att.shape)
    scores_all = source2attr @ att2att @ target2attr.T
    print('score_all.shape: ', scores_all.shape)
    print('score_all.max()', scores_all.max())
    print('score_all.min()', scores_all.min())
    scores_all = (scores_all - np.min(scores_all)) / (np.max(scores_all) - np.min(scores_all))

    ill_pair = ill_pair_load(d)
    scores = np.zeros((len(ill_pair), len(ill_pair)), dtype=np.float32)
    ill_pair = np.array(ill_pair)
    for i, l in enumerate(ill_pair[:, 0]):
        for j, r in enumerate(ill_pair[:, 1]):
            scores[i][j] = scores_all[l][r - len(id1_attr)]

    print(scores.shape)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    # np.save(f'../data/DBP15K/DBP_ECS_1/Att_{d}_6.npy', scores)  # ill_pair & Roberta & translate & att2att
    # np.save(f'../data/DBP15K/DBP_ECS_1/Att_{d}_7.npy', scores)  # ill_pair & Roberta & no_translate & att2att 搞错了
    # np.save(f'../data/DBP15K/DBP_ECS_1/Att_{d}_8.npy', scores)  # ill_pair & LaBSE & no_translate & att2att 搞错了
    # np.save(f'../data/DBP15K/DBP_ECS_1/Att_{d}_9.npy', scores)  # ill_pair & LaBSE & no_translate & att2att & 修改了att和id的对应顺序
    np.save(f'../data/DBP15K/DBP_ECS_1/Att_{d}_10.npy', scores)  # ill_pair & roberta & translate & att2att & 修改了att和id的对应顺序


    sinkhorn_test(scores, scores.shape[0], device=device)



"""
Att_6
    ja_en attribute num:6542
    ja_en attribute num:6067
    100%|██████████| 52/52 [00:05<00:00,  9.73it/s]
    100%|██████████| 48/48 [00:04<00:00, 10.73it/s]
    ill pair length:15000
    (15000, 15000)
            total is  15000
            hits@1 is 0.005333333333333333
            hits@5 is 0.013066666666666667
            hits@10 is 0.0198
            MR is 3664.4306640625
            MRR is 0.011395173147320747
    fr_en attribute num:9175
    fr_en attribute num:6423
    100%|██████████| 72/72 [00:11<00:00,  6.43it/s]
    100%|██████████| 51/51 [00:04<00:00, 10.54it/s]
    ill pair length:15000
    (15000, 15000)
            total is  15000
            hits@1 is 0.0009333333333333333
            hits@5 is 0.0034666666666666665
            hits@10 is 0.0061333333333333335
            MR is 4895.33203125
            MRR is 0.0037927969824522734
    zh_en attribute num:8345
    zh_en attribute num:7174
    100%|██████████| 66/66 [00:06<00:00, 10.13it/s]
    100%|██████████| 57/57 [00:05<00:00, 10.36it/s]
    ill pair length:15000
    (15000, 15000)
            total is  15000
            hits@1 is 0.012
            hits@5 is 0.032
            hits@10 is 0.046733333333333335
            MR is 3135.275146484375
            MRR is 0.02515498362481594
"""


"""
Att_10
    zh_en
    zh_en attribute num:7185
    zh_en attribute num:7174
    (19388, 7185)
    (19572, 7174)
    /home/wluyao/.local/lib/python3.8/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()
    100%|██████████| 57/57 [00:10<00:00,  5.57it/s]
    100%|██████████| 57/57 [00:10<00:00,  5.24it/s]
    att2att.shape:  (7185, 7174)
    score_all.shape:  (19388, 19572)
    score_all.max() 18210514.0
    score_all.min() 140.86707
    ill pair length:15000
    (15000, 15000)
            total is  15000
            hits@1 is 0.0015333333333333334
            hits@5 is 0.004733333333333333
            hits@10 is 0.007466666666666667
            MR is 4388.38330078125
            MRR is 0.0044885422103106976
            
"""









