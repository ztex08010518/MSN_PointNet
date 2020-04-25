import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import sys
import numpy as np
from torch.autograd import Variable


sys.path.append("./models/")
from MSN_model import PointNetfeat, PointGenCon 
from PointNet_model import PointNetEncoder, feature_transform_reguliarzer
import expansion_penalty_module as expansion

def norm_feature(feature):
    norm = feature.norm(p=2, dim=1, keepdim=True)
    x_normalized = feature.div(norm)
    # feat = feature.cpu().detach().numpy()
    # feat_norm = []
    # for f in feat:
        #f_norm = (f - np.min(f))/np.ptp(f) # to (0, 1)
    #     f_norm = 2.*(f - np.min(f))/np.ptp(f)-1 # to (-1, 1)
    #     feat_norm.append(f_norm)
    # x_normalized = torch.from_numpy(np.array(feat_norm)).float().cuda()
    return x_normalized

class PMSN_concat_cls(nn.Module):
    def __init__(self, n_class = 40, num_points = 8192, bottleneck_size = 1024, n_primitives = 16, norm_mode = 'none', a = 1, b = 1):
        super(PMSN_concat_cls, self).__init__()

        # PointNet 
        self.PN_encoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)
        
        # MSN
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.norm_mode = norm_mode
        self.a = a
        self.b = b

        self.MSN_encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )

        self.alpha = nn.Parameter(torch.FloatTensor([self.a]))
        self.beta = nn.Parameter(torch.FloatTensor([self.b]))

        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0,self.n_primitives)])
        self.expansion = expansion.expansionPenaltyModule()
        
        # cls
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # get PointNet feature
        # get MSN feature
        GFV = self.MSN_encoder(x) # GFV: (B, 1024)
        
        PN_feature, trans, trans_feat = self.PN_encoder(x) # x: (B, 1024)
        if self.norm_mode == 'norm':
            PN_feature = norm_feature(PN_feature)
            GFV = norm_feature(GFV)
            # print("Normalize both PN and GFV features")
        elif self.norm_mode == 'ab':
            PN_feature = PN_feature * self.a
            GFV = GFV * self.b
            # print("PN features times {}, GFV features times {}".format(self.a, self.b))
        elif self.norm_mode == 'learn':
            PN_feature = PN_feature * self.alpha
            GFV = GFV * self.beta
            # PN_feature = norm_feature(PN_feature) * self.alpha
            # GFV = norm_feature(GFV) * self.beta
            print("alpha: {}, beta: {}".format(self.alpha.item(), self.beta.item())) 
        else:
            assert self.norm_mode == 'none'
            # print("No feature normalization")
        
        # concatenate feature
        x = torch.cat((PN_feature, GFV), 1)
        concat_feat = x

        # reconstruct coarse output
        outs = []
        for i in range(0,self.n_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(GFV.size(0), 2, self.num_points//self.n_primitives)) # (B, 2, 512)
            rand_grid.data.uniform_(0,1)
            patch = GFV.unsqueeze(2).expand(GFV.size(0), GFV.size(1), rand_grid.size(2)).contiguous() # (B, 1024, 512)
            patch = torch.cat( (rand_grid, patch), 1).contiguous()
            outs.append(self.decoder[i](patch))

        outs = torch.cat(outs,2).contiguous() 
        coarse_out = outs.transpose(1, 2).contiguous()

        # expansion loss
        dist, _, mean_mst_dis = self.expansion(coarse_out, self.num_points//self.n_primitives, 1.5)
        loss_mst = torch.mean(dist) # mst: minimum spanning tree
         
        # cls loss
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        pred = F.log_softmax(x, dim=1)
        
        #### no expansion loss ####
        #loss_mst = 0

        return pred, trans_feat, loss_mst, coarse_out, PN_feature, GFV, concat_feat

class PMSN_pretrain_cls(nn.Module):
    def __init__(self, n_class = 40, num_points = 8192, bottleneck_size = 1024, n_primitives = 16, norm_mode = 'none'):
        super(PMSN_pretrain_cls, self).__init__()

        # MSN
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.norm_mode = norm_mode
        self.MSN_encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0,self.n_primitives)])
        self.expansion = expansion.expansionPenaltyModule()
        
        # cls
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # get MSN feature
    
        GFV = self.MSN_encoder(x) # GFV: (B, 1024)
        
        if self.norm_mode == 'norm':
            GFV = norm_feature(GFV)
            # print("Normalize GFV feature")
        else:
            assert self.norm_mode == 'none'
            # print("No feature normalization")
            

        # reconstruct coarse output
        outs = []
        for i in range(0,self.n_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(GFV.size(0), 2, self.num_points//self.n_primitives)) # (B, 2, 512)
            rand_grid.data.uniform_(0,1)
            patch = GFV.unsqueeze(2).expand(GFV.size(0), GFV.size(1), rand_grid.size(2)).contiguous() # (B, 1024, 512)
            patch = torch.cat( (rand_grid, patch), 1).contiguous()
            outs.append(self.decoder[i](patch))

        outs = torch.cat(outs,2).contiguous() 
        coarse_out = outs.transpose(1, 2).contiguous()

        # expansion loss
        dist, _, mean_mst_dis = self.expansion(coarse_out, self.num_points//self.n_primitives, 1.5)
        loss_mst = torch.mean(dist) # mst: minimum spanning tree
        

        # cls loss
        x = F.relu(self.bn1(self.fc1(GFV)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        pred = F.log_softmax(x, dim=1)
        
    
        #### no expansion loss ####
        #loss_mst = 0
        
        return pred, None, loss_mst, coarse_out, None, GFV, GFV


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001, trans_feat=False):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.trans_feat = trans_feat

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        if  self.trans_feat:
            mat_diff_loss = feature_transform_reguliarzer(trans_feat)
            total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
            return total_loss
        else:
            return loss

class PointNet_cls(nn.Module):
    def __init__(self, n_class = 40, num_points = 8192, bottleneck_size = 1024, n_primitives = 16):
        super(PointNet_cls, self).__init__()

        # PointNet 
        self.PN_encoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)
        
        
        # cls
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # get PointNet feature
        
        x, trans, trans_feat = self.PN_encoder(x) # x: (B, 1024)
        
        # cls loss
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        pred = F.log_softmax(x, dim=1)
        
        #### no expansion loss ####
        loss_mst = 0
        coarse_out = None

        return pred, trans_feat, loss_mst, coarse_out
