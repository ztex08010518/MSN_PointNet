import open3d as o3d
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import random
import warnings

from utils import  *

warnings.filterwarnings('ignore')
root_dirs = {"ModelNet": "/eva_data/psa/datasets/PointNet/ModelNet40_pcd",
            "ShapeNet": "/eva_data/psa/datasets/MSN_PointNet/ShapeNetCore.v1"}
obj_name_files = {"ModelNet": "modelnet40_shape_names.txt",
                  "ShapeNet": "ShapeNetCore.v1_ID.txt"}
class DataLoader(Dataset):
    def __init__(self, root, dataset="ModelNet", sparsify_mode="PN",  npoint=1024, split='train', cache_size=15000):
        self.root = root
        self.npoints = npoint

        self.catfile = os.path.join(self.root, obj_name_files[dataset])

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.sparsify_mode = sparsify_mode
        print("Input data mode is "+ self.sparsify_mode)        

        assert (split == 'train' or split == 'test')
        self.datapath = []
        for class_name in self.cat:
            file_dir = os.path.join(self.root, class_name, split)
            files = os.listdir(file_dir)
            for filename in files:
                file_path = os.path.join(file_dir, filename)
                self.datapath.append((class_name, file_path))

        '''
        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], split, shape_ids[split][i]) + '.pcd') for i
                         in range(len(shape_ids[split]))]
        '''
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)

            ''' 
            # read txt file format
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            '''
            # read pcd file format
            point_set = o3d.io.read_point_cloud(fn[1])
            point_set = np.array(point_set.points)
            
            # 取 points subset
            #point_set = point_set[0:self.npoints,:]
            
            # points normalize
            points = pcd_normalize(point_set)
            # points /= 2 #########################################

            #if len(self.cache) < self.cache_size:
            #    self.cache[index] = (point_set, cls)
        
            if self.sparsify_mode == "PN":
                    points = points[:self.npoints, :]
                    points = resample_pcd(points, self.npoints)
                    # print(points.shape)
            elif self.sparsify_mode == "random":
                points = random_sample(points, self.npoints)
            elif self.sparsify_mode == "fps":
                points = farthest_point_sample(points, self.npoints)
            elif "zorder" in self.sparsify_mode:
                z_values = get_z_values(points) # Get z values for all points
                points_zorder = points[np.argsort(z_values)] # Sort input points with z values

                if self.sparsify_mode == "zorder":
                    points = keep_zorder(points_zorder)
                    #print(points.shape)
                elif self.sparsify_mode == "multizorder":
                    points = keep_multizorder(points_zorder)
                else:
                    assert False, "You should choose [zorder] or [multizorder]"
            else:
                assert False, "PLZ verify sparsify mode is in [PN, random, fps, zorder, multizorder] or not"


        return points.astype(np.float32), cls
        #return points, cls
