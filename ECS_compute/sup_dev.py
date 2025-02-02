
path = r'/home/wluyao/EIEA/data/MMEA-data/seed0.5/FB15K-YAGO15K/ref_ent_ids'
# path = r'/home/wluyao/EIEA/data/MMEA-data/seed0.5/FB15K-DB15K/links'
# path = r'/home/wluyao/EIEA/data/MMEA-data/seed0.8/FB15K-DB15K/links'

alignment_pair = []
with open(path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        e1, e2 = line.split()
        alignment_pair.append((int(e1), int(e2)))


path = r'/home/wluyao/EIEA/data/MMEA-data/seed0.5/FB15K-YAGO15K'
# path = '/home/wluyao/EIEA/data/MMEA-data/seed0.5/FB15K-DB15K'
# path = '/home/wluyao/EIEA/data/MMEA-data/seed0.8/FB15K-DB15K'
sup_path = path + '/sup_ent_ids'
dev_path = path + '/dev_ent_ids'

half = len(alignment_pair) // 2
new_sup_pairs = alignment_pair[:half]
new_dev_pairs = alignment_pair[half:]

# Write the new sup file
with open(sup_path, 'w') as file:
    for e1, e2 in new_sup_pairs:
        file.write(f"{e1} {e2}\n")

# Write the new dev file
with open(dev_path, 'w') as file:
    for e1, e2 in new_dev_pairs:
        file.write(f"{e1} {e2}\n")


