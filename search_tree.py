from annoy import AnnoyIndex
from utils.load_data import load_user_data, str2np
import random
import time

time_ = time.time()
tree = AnnoyIndex(512, 'euclidean')
user_data, user_info = load_user_data()
for idx, user in enumerate(user_data):
    feat = user[3]
    tree.add_item(idx, feat.tolist())

tree.build(100)
print((time.time() - time_)*1000)
tree.save('src/face_embedding.ann')


# f = 2  # Length of item vector that will be indexed
# metric = 'euclidean'
# t = AnnoyIndex(f, metric)
# l = []
# for i in range(1000):
#     v = [i, i]
#     t.add_item(i, v)
#     l.append(v)
#
# t.build(10) # 10 trees
# t.save('test.ann')
# # ...
# u = AnnoyIndex(f, metric)
# u.load('test.ann') # super fast, will just mmap the file
# near = u.get_nns_by_vector([10.0, 8.0], 1) # will find the 1000 nearest neighbors
# print(near)
# print(u.get_item_vector(near[0]))