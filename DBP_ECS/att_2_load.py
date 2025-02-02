from tqdm import tqdm


def ent_load(dataset):
    ent_1 = {}
    with open('../data/DBP15K/DBP_1/{}/ent_ids_1'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_name = line.split('\t')
            ent_1[int(ent_id)] = ent_name
    ent_2 = {}
    with open('../data/DBP15K/DBP_1/{}/ent_ids_2'.format(dataset), 'r') as f:
        for line in f:
            line = line.strip()
            ent_id, ent_name = line.split('\t')
            ent_2[int(ent_id)] = ent_name

    return ent_1, ent_2

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



dataset = ['ja_en', 'fr_en']
for d in dataset:
    print(d)
    ent_1, ent_2 = ent_load(d)
    att_1, att_2 = att_load_1(d)
    src_lang, tgt_lang = d.split('_')
    id_att_2 = {}
    for id_2 in tqdm(ent_2.keys()):
        if ent_2[id_2] in att_2.keys():
            att_list = att_2[ent_2[id_2]]
            id_att_2[id_2] = att_list
        else:
            id_att_2[id_2] = ['NO_property']
    with open(f'../data/attr_trans/{d}_id_att_2', 'w') as f:
        for k,v in id_att_2.items():
            f.write(f'{k}\t{" ".join(v)}\n')