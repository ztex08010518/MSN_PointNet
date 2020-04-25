import os
import torch

from os import listdir

# path = "/eva_data/psa/code/outputs/PointNet/ModelNet40_cls/PN/checkpoints/"
path = "/eva_data/psa/code/outputs/MSN_PointNet/concat/ShapeNet_all/PN/fix_both/zorder1024_normalize_norm/weights/"

weights = sorted(listdir(path))
# test = [weights[0], weights[1], weights[-1]]
test = weights


for w_f in test:
    print(w_f)
    weight = torch.load(os.path.join(path, w_f))
    
    # print(weight['model_state_dict'].keys())
    print(weight['model_state_dict']['PN_encoder.stn.conv1.weight'][0][0][0])
    # print(weight['model_state_dict']['feat.stn.conv1.weight'][0][0][0])
    # print(weight['model_state_dict']['MSN_encoder.0.bn2.weight'][0])
