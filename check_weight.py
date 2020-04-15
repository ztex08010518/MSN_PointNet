import os
import torch

from os import listdir

path = "/eva_data/psa/code/outputs/MSN_PointNet/concat/open_PN/MSN_norm/weights/"
weights = sorted(listdir("/eva_data/psa/code/outputs/MSN_PointNet/concat/open_PN/MSN_norm/weights"))
test = [weights[0], weights[1], weights[-1]]

for w_f in test:
    print(w_f)
    weight = torch.load(os.path.join(path, w_f))
    
    # print(weight['model_state_dict'].keys())
    # print(weight['model_state_dict']['PN_encoder.stn.conv1.weight'])
    print(weight['model_state_dict']['MSN_encoder.0.bn2.weight'])
