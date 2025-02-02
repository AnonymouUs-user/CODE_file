import torch
from sentence_transformers import SentenceTransformer
from pkl_read import ent_load, dev_pair_load, sinkhorn_test, ill_pair_load
import numpy as np
from tqdm import trange, tqdm
import requests
import json
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBart50Tokenizer


def load_att(dataset):
    d_l = dataset.split('_')[0]
    ent1_att_path = '../data/DBP15K/{}/training_attrs_1'.format(dataset)
    ent2_att_path = '../data/DBP15K/{}/training_attrs_2'.format(dataset)
    att_1 ={}
    att_2 ={}

    with open(ent1_att_path, 'r') as f:
        for line in f:
            line = line.strip()
            l = line.split('\t')
            l_p = 'http://' + str(d_l) +'.dbpedia.org/property/'
            new_l = [item.replace("http://dbpedia.org/property/", "").replace(l_p, "") for item in l[1:]]
            att_1[l[0]] = ' '.join(new_l)
    with open(ent2_att_path, 'r') as f:
        for line in f:
            line = line.strip()
            l = line.split('\t')
            # new_l = [item.replace("http://ja.dbpedia.org/property/", "") for item in l[1:]]
            l_p = 'http://' + str(d_l) +'.dbpedia.org/property/'
            new_l = [item.replace("http://dbpedia.org/property/", "").replace(l_p, "") for item in l[1:]]
            att_2[l[0]] = '||'.join(new_l)
    return att_1, att_2

def translate_1(sentence, src_lan, tgt_lan):
    apikey = '9a747968c998ccf465d8b3764288facd'
    url = 'http://api.niutrans.com/NiuTransServer/translation?'
    data = {"from": src_lan, "to": tgt_lan, "apikey": apikey, "src_text": sentence}
    res = requests.post(url, data=data)
    res_dict = json.loads(res.text)
    if "tgt_text" in res_dict:
        result = res_dict['tgt_text']
    else:
        result = res
    return result


def id_att(ent_1, ent_2, att_1, att_2):
    id_att_1 = {}
    id_att_2 = {}

    for id in ent_1.keys():
        if ent_1[id] in att_1.keys():
            att = att_1[ent_1[id]]
            # att_t = translate(att, s_lan, t_lan)
            id_att_1[id] = att
            # att_t : 'Capacity Construction Stadium Name Ground Construction Cost Opening Team, Tournament Image Pitch Size'
            # id_att_1[id] = att_t
        else:
            id_att_1[id] = 'No_property'
    for id in ent_2.keys():
        if ent_2[id] in att_2.keys():
            att = att_2[ent_2[id]]
            id_att_2[id] = att
        else:
            id_att_2[id] = 'No_property'
    return id_att_1, id_att_2


def translate(sentences, src_lan, trans_model, trans_tokenizer, device):

    if src_lan == 'zh':
        src_lang = 'zh_CN'
    elif src_lan == 'ja':
        src_lang = 'ja_XX'
    else:
        src_lang = 'fr_XX'

    target_lang = "en_XX"

    # model = MBartForConditionalGeneration.from_pretrained("../mbart_large_50_many_to_many_mmt")
    # tokenizer =  MBart50Tokenizer.from_pretrained("../mbart_large_50_many_to_many_mmt")
    # model.to(device)
    # model.eval()
    translated_texts = []
    trans_tokenizer.src_lang = src_lang

    batch_size = 16
    for i in tqdm(range(0, len(sentences), batch_size), desc="Translating", unit="batch"):
        batch_sentences = sentences[i:i + batch_size]
        encoded_input = trans_tokenizer(batch_sentences,
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True,
                                  max_length=512).to(device)
        # 没有加max_length，不进行任何截断
        generated_tokens = trans_model.generate(**encoded_input,
                                          forced_bos_token_id=trans_tokenizer.lang_code_to_id[target_lang])
        result = trans_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_texts.extend(result)

    return translated_texts


dataset = ['ja_en', 'fr_en', 'zh_en']
# dataset = ['fr_en', 'zh_en']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trans_model = MBartForConditionalGeneration.from_pretrained('../mbart_large_50_many_to_many_mmt')
trans_tokenizer = MBart50Tokenizer.from_pretrained('../mbart_large_50_many_to_many_mmt')
trans_model.to(device)
trans_model.eval()

# 将翻译属性保存下来
for d in dataset:
    print(d)
    att_1, att_2 = load_att(d)
    ent_1, ent_2 = ent_load(d)
    s_lan, t_lan = d.split('_')
    id_att_1, id_att_2 = id_att(ent_1, ent_2, att_1, att_2)
    att_lt_1_ = list(id_att_1.values())
    print("tranlate process .... ")
    att_lt_1 = translate(att_lt_1_, s_lan, trans_model, trans_tokenizer, device)
    print('translate over!')
    with open(f'../data/attr_trans/{d}_attr_1.txt', 'w', encoding='utf-8') as f:
        for i in att_lt_1:
            f.write(i + '\n')
    with open(f'../data/attr_trans/{d}_attr_2.txt', 'w', encoding='utf-8') as f:
        for i in list(id_att_2.values()):
            f.write(i + '\n')
exit(0)

    # att_lt_2 = list(id_att_2.values())




model = SentenceTransformer(
    '../Roberta_finetuning_semantic_similarity_stsb_multi_mt/').to(device)
# model = SentenceTransformer('../LaBSE/').to(device)
model.eval()



for d in dataset:
    print(d)
    att_1, att_2 = load_att(d)
    ent_1, ent_2 = ent_load(d)
    # dev_pair = dev_pair_load(d)
    ill_pair = ill_pair_load(d)
    # [(0, 19814), (1, 19815), (2, 19816), (3, 19817), (4, 19818)]
    # ent_1[0] 'http://ja.dbpedia.org/resource/アリアンツ・リヴィエラ'
    # ent_2[19814] 'http://dbpedia.org/resource/Allianz_Riviera'
    # att_1[ent_1[0]] '収容能力||起工||スタジアム名称||グラウンド||建設費||開場||使用チーム、大会||画像||ピッチサイズ'
    #   对应的翻译结果：Capacity||Groundbreaking||Name of stadium||Grounds||Construction cost||Opening||Teams, tournaments used||Image|||Pitch size'
    # att_2[ent_2[19814]] 'http://xmlns.com/foaf/0.1/name||cost||capacity||name||brokeGround||owner||surface||caption||logoImage'

    ent_1_num = len(ent_1)
    ent_2_num = len(ent_2)
    s_lan, t_lan = d.split('_')
    id_att_1, id_att_2 = id_att(ent_1, ent_2, att_1, att_2)

    att_lt_1_ = list(id_att_1.values())
    print("tranlate process .... ")
    att_lt_1 = translate(att_lt_1_, s_lan, trans_model, trans_tokenizer, device)
    print('translate over!')
    att_lt_2 = list(id_att_2.values())
    # batch_size = 512
    batch_size = 1024
    att_embedding_1 = []
    att_embedding_2 = []

    for i in trange(0, ent_1_num, batch_size):
        key_sents = att_lt_1[i:i + batch_size]
        att_embedding_1.append(model.encode(key_sents))
        # print(att_embedding_1[-1].shape)
    att_embedding_1 = np.concatenate(att_embedding_1, axis=0)

    for i in trange(0, ent_2_num, batch_size):
        key_sents = att_lt_2[i:i + batch_size]
        att_embedding_2.append(model.encode(key_sents))
    att_embedding_2 = np.concatenate(att_embedding_2, axis=0)

    print(att_embedding_1.shape)
    print(att_embedding_2.shape)

    scores_all = np.matmul(att_embedding_1, att_embedding_2.T)
    print(scores_all.shape)
    scores_all = (scores_all - np.min(scores_all)) / (np.max(scores_all) - np.min(scores_all))
    print(scores_all[0][0])

    # scores = np.zeros((len(dev_pair), len(dev_pair)), dtype=np.float32)
    # dev_pair = np.array(dev_pair)
    # for i, l in enumerate(dev_pair[:, 0]):
    #     for j, r in enumerate(dev_pair[:, 1]):
    #         scores[i][j] = scores_all[l][r-len(ent_1)]

    scores = np.zeros((len(ill_pair), len(ill_pair)), dtype=np.float32)
    ill_pair = np.array(ill_pair)
    for i, l in enumerate(ill_pair[:, 0]):
        for j, r in enumerate(ill_pair[:, 1]):
            scores[i][j] = scores_all[l][r - len(ent_1)]

    print(scores.shape)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    # save the scores as .npy file
    # np.save(f'../data/DBP15K/DBP_ECS/Att_{d}_1.npy', scores) # Robert
    # np.save(f'../data/DBP15K/DBP_ECS/Att_{d}_3.npy', scores) # LaBSE
    # np.save(f'../data/DBP15K/DBP_ECS_1/Att_{d}_3.npy', scores)  # ill_pair & LaBSE
    # np.save(f'../data/DBP15K/DBP_ECS_1/Att_{d}_4.npy', scores)  # ill_pair & LaBSE & translate
    np.save(f'../data/DBP15K/DBP_ECS_1/Att_{d}_5.npy', scores)  # ill_pair & Roberta & translate

    sinkhorn_test(scores, scores.shape[0], device=device)



"""
1.
    ja_en
    dev pair length:10500
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [06:52<00:00, 10.57s/it]
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [05:59<00:00,  9.22s/it]
    (19814, 1024)
    (19780, 1024)
    (19814, 19780)
    (10500, 10500)
                    total is  10500
                    hits@1 is 0.019714285714285715
                    hits@5 is 0.05104761904761905
                    hits@10 is 0.076
                    MR is 2110.478759765625
                    MRR is 0.040488358587026596
    fr_en
    dev pair length:10500
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [05:45<00:00,  8.86s/it]
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [06:35<00:00,  9.89s/it]
    (19661, 1024)
    (19993, 1024)
    (19661, 19993)
    (10500, 10500)
                    total is  10500
                    hits@1 is 0.0015238095238095239
                    hits@5 is 0.007523809523809524
                    hits@10 is 0.014095238095238095
                    MR is 2877.2216796875
                    MRR is 0.007567377761006355
    zh_en
    dev pair length:10500
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 38/38 [07:14<00:00, 11.44s/it]
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [06:10<00:00,  9.49s/it]
    (19388, 1024)
    (19572, 1024)
    (19388, 19572)
    (10500, 10500)
                    total is  10500
                    hits@1 is 0.06495238095238096
                    hits@5 is 0.14466666666666667
                    hits@10 is 0.19247619047619047
                    MR is 1268.18505859375
                    MRR is 0.10811395198106766


2. LaBSE
    ja_en
        dev pair length:10500
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:41<00:00,  1.07s/it]
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:39<00:00,  1.00s/it]
        (19814, 768)
        (19780, 768)
        (19814, 19780)
        0.5599729
        (10500, 10500)
                        total is  10500
                        hits@1 is 0.04780952380952381
                        hits@5 is 0.11552380952380953
                        hits@10 is 0.16352380952380952
                        MR is 547.0552368164062
                        MRR is 0.0889933854341507
    fr_en
        dev pair length:10500
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:33<00:00,  1.16it/s]
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:53<00:00,  1.35s/it]
        (19661, 768)
        (19993, 768)
        (19661, 19993)
        0.79370373
        (10500, 10500)
                        total is  10500
                        hits@1 is 0.011333333333333334
                        hits@5 is 0.04057142857142857
                        hits@10 is 0.06676190476190476
                        MR is 1092.2547607421875
                        MRR is 0.03248130530118942
    zh_en
        dev pair length:10500
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 38/38 [00:41<00:00,  1.10s/it]
        100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:38<00:00,  1.02it/s]
        (19388, 768)
        (19572, 768)
        (19388, 19572)
        0.738148
        (10500, 10500)
                        total is  10500
                        hits@1 is 0.1180952380952381
                        hits@5 is 0.2321904761904762
                        hits@10 is 0.2896190476190476
                        MR is 330.5966796875
                        MRR is 0.1775858998298645

"""

"""
ja_en
ill pair length:15000
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:39<00:00,  1.02s/it]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:36<00:00,  1.06it/s]
(19814, 768)
(19780, 768)
(19814, 19780)
0.5599729
(15000, 15000)
                total is  15000
                hits@1 is 0.03886666666666667
                hits@5 is 0.0944
                hits@10 is 0.13306666666666667
                MR is 771.6363525390625
                MRR is 0.07338869571685791
fr_en
ill pair length:15000
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:32<00:00,  1.19it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.04it/s]
(19661, 768)
(19993, 768)
(19661, 19993)
0.79370373
(15000, 15000)
                total is  15000
                hits@1 is 0.0084
                hits@5 is 0.0298
                hits@10 is 0.0508
                MR is 1568.1951904296875
                MRR is 0.02512402832508087
zh_en
ill pair length:15000
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 38/38 [00:40<00:00,  1.06s/it]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:38<00:00,  1.02it/s]
(19388, 768)
(19572, 768)
(19388, 19572)
0.738148
(15000, 15000)
                total is  15000
                hits@1 is 0.1046
                hits@5 is 0.20326666666666668
                hits@10 is 0.26026666666666665
                MR is 480.66534423828125
                MRR is 0.15757039189338684

"""


"""
labse:
ja_en
ill pair length:15000
tranlate process .... 
Translating:   0%|                                                                                                                                         Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Translating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
translate over!
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
(19814, 768)
(19780, 768)
(19814, 19780)
0.51696545
(15000, 15000)
                total is  15000
                hits@1 is 0.033133333333333334
                hits@5 is 0.08533333333333333
                hits@10 is 0.122
                MR is 839.6307983398438
                MRR is 0.06564650684595108

"""

"""
Roberta
fr_en
ill pair length:15000
tranlate process .... 
Translating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1229/1229 [1:35:29<00:00,  4.66s/batch]
translate over!
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [01:33<00:00,  4.70s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [02:46<00:00,  8.33s/it]
(19661, 1024)
(19993, 1024)
(19661, 19993)
0.5493406
(15000, 15000)
                total is  15000
                hits@1 is 0.0019333333333333333
                hits@5 is 0.006933333333333333
                hits@10 is 0.012933333333333333
                MR is 3828.995361328125
                MRR is 0.00727814668789506
zh_en
ill pair length:15000
tranlate process .... 
Translating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1212/1212 [1:39:29<00:00,  4.93s/batch]
translate over!
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19/19 [01:20<00:00,  4.22s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [02:42<00:00,  8.11s/it]
(19388, 1024)
(19572, 1024)
(19388, 19572)
0.58820724
(15000, 15000)
                total is  15000
                hits@1 is 0.03186666666666667
                hits@5 is 0.0826
                hits@10 is 0.11453333333333333
                MR is 1781.591064453125
                MRR is 0.06135212630033493
"""