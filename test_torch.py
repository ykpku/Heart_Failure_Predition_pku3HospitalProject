import torch

# # 假设batch_rule_tensor的规模是:Ne*Ne*Nr，首先将其扩展成四维:Nb*Ne*Ne*Nr。 其中，Nb,Ne,Nr 分别代表batch大小，实体种类，关系种类
# batch_rule_tensor_rep = batch_rule_tensor.unsqueeze(0).repeat(Nb, 1, 1, 1)
#
# # 假设batch_diag_ent_indices的规模：Nb*Nc，要先扩展成四维：Nb*Nc*Ne*Nr
# batch_diag_ent_indices_rep1 = batch_diag_ent_indices.unsqueeze(2).unsqueeze(3).repeat(1, 1, Ne, Nr)
# # 假设batch_diag_ent_indices的规模：Nb*Nc，要先扩展成四维：Nb*Nc*Nc*Nr
# batch_diag_ent_indices_rep2 = batch_diag_ent_indices.unsqueeze(1).unsqueeze(3).repeat(1, Nc, 1, Nr)
#
# # 两次gather
# T = batch_rule_tensor_rep.gather(1, batch_diag_ent_indices_rep1).gather(2, batch_diag_ent_indices_rep2)

# E = torch.randn(2, 3)
# print(E, E.shape)
#
# Er1 = E.unsqueeze(1).unsqueeze(3).repeat(1, 4, 1, 5)
# print(Er1, Er1.shape)
# E = E.requires_grad_().long()
# print(E.type())
# print(E.requires_grad)

import pandas as pd
import numpy as np
from copy import deepcopy
data1 = pd.DataFrame(np.array([[1, 2, 3], [2, 4, 6], [6, 4, 5]]), columns=['sdf', 'sf', 'sefwef'])
new_columns = [i for i in data1.columns if i != 'sdf']
new_columns.extend(['q1', 'q2', 'q3'])
print(data1)

data_q1 = deepcopy(data1['sdf'])
data_q1[data_q1 < 1] = 0
data_q1[data_q1 >= 1] = 1
print(data_q1)

data_q2 = deepcopy(data1['sdf'])
data_q2[data_q2 < 2] = 0
data_q2[data_q2 >= 2] = 1

data_q3 = deepcopy(data1['sdf'])
print(data_q2)
data_q3[data_q3 < 6] = 0
data_q3[data_q3 >= 6] = 1
print(data_q3)

data1.drop('sdf', axis=1, inplace=True)

sdf = pd.concat([data1, data_q1, data_q2, data_q3], axis=1, ignore_index=True)
sdf.columns = new_columns
print(sdf)
print(new_columns)
# print(sdf)
# sdf[sdf['saa'] < 2] = 0
# sdf[sdf['saa'] >= 2] = 1
# print(data1)
# print(sdf)