import numpy as np
import os
import torch

from os import listdir
from libsvm.python.svmutil import *
from libsvm.python.svm import *


root = "/eva_data/psa/code/outputs/MSN_PointNet/concat/ShapeNet_all/PN/open/zorder1024_normalizelearn/features"

files = sorted(listdir(root))
#print(files)

test_id = '5'

x = np.load(root+'/'+test_id+"_features.npy", allow_pickle=True)
y = np.load(root+'/'+test_id+"_gt.npy", allow_pickle=True)

training_data = []
gt = []
for batch_x, batch_y in zip(x, y):
    # print(type(batch_y))
    training_data.extend(batch_x)
    gt.extend(batch_y.cpu().data.numpy())
training_data = np.array(training_data)

training_data = [{idx: f for idx, f in enumerate(features)} for features in training_data]

# print("Ground truth: ", gt)
# print("Training data: ", training_data)
# print(training_data.shape)
print("Size of data", len(training_data))

print(gt[:100])
m = svm_train(gt[:200], training_data[:200], '-c 4')

p_label, p_acc, p_val = svm_predict(gt[:200], training_data[:200], m)

