from os.path import join as pjoin
import re
import numpy as np


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


ent2id = {}
id2ent = {}
source2id = {}
id2source = {}
target2id = {}
id2target = {}
train_pair = []
dev_pair = []
id2attrs = []
seeds = []
source_keys = set()
target_keys = set()

with open(pjoin('../data/ent/FB-DB/', 'ent_ids_1'), 'r') as f:
    for line in f:
        line = line.strip()
        ent_id, ent_name = line.split('\t')
        ent2id[ent_name] = int(ent_id)
        id2ent[int(ent_id)] = ent_name
        source2id[ent_name] = int(ent_id)
        id2source[int(ent_id)] = ent_name

with open(pjoin('../data/ent/FB-DB/', 'ent_ids_2'), 'r') as f:
    for line in f:
        line = line.strip()
        ent_id, ent_name = line.split('\t')
        ent2id[ent_name] = int(ent_id)
        id2ent[int(ent_id)] = ent_name
        target2id[ent_name] = int(ent_id) - len(source2id)  # 从0开始计数
        id2target[int(ent_id) - len(source2id)] = ent_name

# with open(pjoin('../data/MMEA-data/seed0.8', 'FB15K-DB15K', 'ref_ent_ids'), 'r') as f:
#     for line in f:
#         line = line.strip()
#         ent_id1, ent_id2 = line.split('\t')
#         seeds.append((int(ent_id1), int(ent_id2)))

with open(pjoin('../data/MMEA-data/seed0.8', 'FB15K-DB15K', 'sup_ent_ids'), 'r') as f:
    for line in f:
        line = line.strip()
        ent_id1, ent_id2 = line.split('\t')
        train_pair.append((int(ent_id1), int(ent_id2)))

with open(pjoin('../data/MMEA-data/seed0.8', 'FB15K-DB15K', 'dev_ent_ids'), 'r') as f:
    for line in f:
        line = line.strip()
        ent_id1, ent_id2 = line.split('\t')
        dev_pair.append((int(ent_id1), int(ent_id2)))


# with open(pjoin('../data/MMEA-data/seed0.8', 'FB15K-DB15K', 'attr'), 'r') as f:
#     try:
#         for i, line in enumerate(f):
#             line = line.strip()
#             tmp = []
#             if i < len(source2id):
#                 attr_value_schemas = line.split('^^^')
#                 for attr_value_schema in attr_value_schemas:
#                     if attr_value_schema == '':
#                         continue
#                     attr, value, schema = attr_value_schema.split('|||')  # KG1 schema
#                     tmp.append((attr, value, schema))
#                 id2attrs.append(tmp)
#             else:
#                 attr_values = line.split('^^^')
#                 for attr_value in attr_values:
#                     if attr_value == '':
#                         continue
#                     attr, value = attr_value.split('|||')  # KG2 schema
#                     tmp.append((attr, value))
#                 id2attrs.append(tmp)
#     except:
#         print('Attr error!')

print(len(dev_pair))
print(len(train_pair))


# source_attr_value_set = set()
# target_attr_value_set = set()
#
# for i, attrs in enumerate(id2attrs):
#     if i < len(source2id):
#         for attr, value, schema in attrs:
#             value = date2float(value)
#             source_attr_value_set.add(attr + ' ' + value)
#             source_keys.add(attr)
#     else:
#         for attr, value in attrs:
#             value = date2float(value)
#             target_attr_value_set.add(attr + ' ' + value)
#             target_keys.add(attr)
#
# print('source attr num:', len(source_keys))  # len(source_keys): 225
# print('target attr num:', len(target_keys))  # len(target_keys): 104
#
# source_attr_value_2_id = {}
# target_attr_value_2_id = {}
# source_id2_attr_value = {}
# target_id2_attr_value = {}
# print('source attr value num:', len(source_attr_value_set))  # source attr value num: 25796
# print('target attr value num:', len(target_attr_value_set))  # target attr value num: 17134
#
# source2attr = np.zeros((len(source2id), len(source_attr_value_set)), dtype=np.float32)
# target2attr = np.zeros((len(target2id), len(target_attr_value_set)), dtype=np.float32)
#
# source_attr_value_list = sorted(list(source_attr_value_set))
# target_attr_value_list = sorted(list(target_attr_value_set))
#
# for i, attrs in enumerate(id2attrs):
#     if i < len(source2id):
#         for attr, value, schema in attrs:
#             value = date2float(value)
#             pos = source_attr_value_list.index(attr + ' ' + value)
#             source2attr[i][pos] = 1
#     else:
#         for attr, value in attrs:
#             value = date2float(value)
#             pos = target_attr_value_list.index(attr + ' ' + value)
#             target2attr[i - len(source2id)][pos] = 1








