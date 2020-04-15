import argparse
import random
import numpy as np
import open3d as o3d
import torch
import torch.optim as optim
import sys
import os
import time, datetime
import logging


sys.path.append("/eva_data/psa/code/data_utils")
from PN_utils import *

sys.path.append("/eva_data/psa/code/data_utils/expansion_penalty")
import expansion_penalty_module as expansion

from dataset import DataLoader
from models.PMSN_cls import *
from utils import *
from pathlib import Path
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='batch size in training [default: 24]')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
parser.add_argument('--gpu', type=str, required=True,  help='specify gpu device [default: 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
parser.add_argument('--sparsify_mode', type=str, required=True, choices=['PN', 'random', 'fps', 'zorder', 'multizorder'], default='PN', help='PointNet input')
parser.add_argument('--MSN_mode', type=str, required=True, choices=['MSN', 'random', 'fps', 'zorder', 'multizorder'], default='random', help='MSN trained on')
parser.add_argument('--method', type=str, default = 'pretrain', choices = ['pretrain', 'concat', 'PointNet'],required=True,  help='Which method to train')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of surface elements')
parser.add_argument('--output_dir', type=str, default='/eva_data/psa/code/outputs/MSN_PointNet',  help='root output dir for everything')
parser.add_argument('--weight_dir', default= '', type=str, help='using which pretrained weight')
parser.add_argument('--fix_mode', type=str, choices = ['open', 'fix_both', 'fix_MSN', 'fix_point'], required=True, help='pretrain-fix encoder, concat-fix both encoder, or fix MSN encoder')
parser.add_argument('--dataset', type=str,default = 'ModelNet', choices = ['Modelnet', 'ShapeNet'], help='to choose input dataset' )

opt = parser.parse_args()
print (opt)

record_save = {
    "train":{
        "instance_acc":[],
        "expansion_loss": [],
        "cls_loss": [],
        "total_loss": []
    },
    "test":{
        "instance_acc":[],
        "class_acc":[]
    }
}


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _, _, _, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)

    # Save testing accuracy (instance and class)
    record_save["test"]["instance_acc"].append(instance_acc)
    record_save["test"]["class_acc"].append(class_acc)

    return instance_acc, class_acc

def save_results(experiment_dir):
    results_dir = os.path.join(experiment_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "train_loss.npy"), record_save)   

if __name__ == "__main__":
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER (GPU)'''
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    '''CREATE DIR'''
    if opt.MSN_mode == 'random':
        idx = opt.weight_dir.index("random")
        opt.MSN_mode += opt.weight_dir[idx+6 : idx+8]

    if opt.MSN_mode in ['zorder', 'multizorder', 'MSN']:
        if opt.weight_dir != '':
            dir_names = opt.weight_dir.split('/')
            opt.MSN_mode =  dir_names[-2] if dir_names[-1] == '' else dir_names[-1]

    #postfix_path = '_'.join([opt.method, opt.MSN_mode])
    
    dir_name = os.path.join(opt.output_dir, opt.method, opt.fix_mode+'_'+opt.sparsify_mode, opt.MSN_mode+'_norm')
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    logname = os.path.join(dir_name, 'log.txt')
    os.system('cp ./train.py %s' % dir_name)
    os.system('cp ./dataset.py %s' % dir_name)
    os.system('cp ./models/PMSN_cls.py %s' % dir_name)


    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = os.path.join(dir_name, "logs")
    os.makedirs(experiment_dir, exist_ok=True)
    
    checkpoints_dir = os.path.join(dir_name,'weights/')
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_dir = os.path.join(experiment_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, opt.method))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(opt)

    '''DATA LOADING'''
    log_string('Load dataset {} ...'.format(opt.dataset))
    ModelNet_PATH = '/eva_data/psa/datasets/PointNet/ModelNet40_pcd/'
    ShapeNet_PATH = '/eva_data/psa/datasets/MSN_PointNet/ShapeNetCore.v1'

    if opt.dataset =='ModelNet':
        TRAIN_DATASET = DataLoader(root=ModelNet_PATH, dataset= opt.dataset, sparsify_mode=opt.sparsify_mode, npoint=opt.num_point, split='train')
        TEST_DATASET = DataLoader(root=ModelNet_PATH, dataset= opt.dataset, sparsify_mode=opt.sparsify_mode, npoint=opt.num_point, split='test')
        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=opt.batch_size, shuffle=True, num_workers=4,drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=opt.batch_size, shuffle=False, num_workers=4,drop_last=True)
    elif opt.dataset == 'ShapeNet':
        TRAIN_DATASET = DataLoader(root=ShapeNet_PATH, dataset= opt.dataset, sparsify_mode=opt.sparsify_mode, npoint=opt.num_point, split='train')
        TEST_DATASET = DataLoader(root=ShapeNet_PATH, dataset= opt.dataset, sparsify_mode=opt.sparsify_mode, npoint=opt.num_point, split='test')
        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=opt.batch_size, shuffle=True, num_workers=4,drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=opt.batch_size, shuffle=False, num_workers=4,drop_last=True)
    print(len(TRAIN_DATASET))
######################################################################
    '''MODEL LOADING'''
    num_class = 40

    if opt.method == 'concat':
        classifier = torch.nn.DataParallel(PMSN_concat_cls(num_class, 8192, 1024, 16)).cuda()
        criterion = get_loss(trans_feat=True).cuda()
        print("===================Concat Mode===========================")
    elif opt.method == 'PointNet':
        classifier = torch.nn.DataParallel(PointNet_cls(num_class, 8192, 1024, 16)).cuda()
        criterion = get_loss(trans_feat=True).cuda()
        print("===================PointNet Mode===========================")
    else:
        classifier = torch.nn.DataParallel(PMSN_pretrain_cls(num_class, 8192, 1024, 16)).cuda()
        criterion = get_loss(trans_feat=False).cuda()
        print("===================Pretrain Mode===========================")
    

    ## load MSN
    try:
        if opt.MSN_mode == "MSN":
            if opt.weight_dir == '':
                print("Use MSN pretrain weight")
                save_model = torch.load('/eva_data/psa/code/MSN/trained_model/network.pth')
            else:
                print("Use MSN {}".format(opt.weight_dir))
                save_model = torch.load(os.path.join(opt.weight_dir,'weights/best_model.pth'))
        elif opt.MSN_mode == 'PointNet':
                print("Use PointNet pretrain in pretrain mode")
        else:
            print("Use {} MSN pretrain weight".format(opt.MSN_mode))
            save_model = torch.load(os.path.join(opt.weight_dir,'weights/best_model.pth'))
        print("load successful")
        model_state_dict = classifier.module.state_dict()
        save_state_dict_en = {'MSN_'+ k: v for k, v in save_model.items() if "encoder" in k}
        save_state_dict_de = {k: v for k, v in save_model.items() if "decoder" in k}
        
        # for e_k, d_k in zip(save_state_dict_en.keys(), save_state_dict_de.keys()):
        #     print("Encoder use pretrained weights: {}".format(e_k))
        #     print("Decoder use pretrained weights: {}".format(d_k))
        # ## change dict 'encoder' to 'MSN_encoder'

        model_state_dict.update(save_state_dict_en)
        model_state_dict.update(save_state_dict_de)
        classifier.module.load_state_dict(model_state_dict)
        log_string('Use MSN pretrain model')
    except Exception:
        log_string('MSN: No existing model, starting training from scratch...')
        print(sys.exc_info())
        start_epoch = 0

    ## load PointNet
    if opt.method != "pretrain":
        try:
            PN_weight = '/eva_data/psa/code/outputs/PointNet/classification_cache/'+opt.sparsify_mode+'/checkpoints/best_model.pth'
            #save_model = torch.load('/eva_data/psa/code/outputs/PointNet/ShapeNet_cls/PN/checkpoints/best_model.pth')
            #save_model = torch.load('/eva_data/psa/code/outputs/PointNet/classification_cache/zorder/checkpoints/best_model.pth')
            print("Loading Pointnet ",opt.sparsify_mode," pretrain successful")
            model_state_dict = classifier.module.state_dict()
            save_state_dict = {k: v for k, v in save_model.items() if "feat" in k}

            model_state_dict.update(save_state_dict)
            classifier.module.load_state_dict(model_state_dict)
            log_string('Use PointNet pretrain model')
        except Exception:
            log_string('PointNet: No existing model, starting training from scratch...')
            print(sys.exc_info())
            start_epoch = 0

    ## Fix model weights
    if opt.fix_mode == 'fix_point':
        print("Fix pretrain PointNet")
        for child in classifier.module.PN_encoder.children():
                for param in child.parameters():
                    param.requires_grad = False
    elif opt.fix_mode != 'open':
        print("Fix MSN")
        for child in classifier.module.MSN_encoder.children():
            for param in child.parameters():
                param.requires_grad = False

        if opt.fix_mode == 'fix_both':
            assert opt.method  == 'concat'
            print("Fix PointNet")
            for child in classifier.module.PN_encoder.children():
                for param in child.parameters():
                    param.requires_grad = False
            

    '''SETUP OPTIMIZER'''
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, classifier.module.parameters()),
            #classifier.module.parameters(),
            lr=opt.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=opt.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, classifier.module.parameters()),
            lr=0.01, 
            momentum=0.9
        )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    all_train_loss = []
    logger.info('Start training...')
    for epoch in range(opt.nepoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, opt.nepoch))

        # TRAIN MODE
        mean_correct = []
        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = random_point_dropout(points)
            points = random_scale_point_cloud(points)
            points = shift_point_cloud(points)
            points = torch.Tensor(points)
            target = target[:, 0]
            
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            # Start training
            classifier.module.train()
            pred, trans_feat, expansion_loss, coarse_out, PN_feature, GFV = classifier(points)
            # print("PointNet: ", PN_feature.shape, type(PN_feature))
            # print(PN_feature.cpu())
            # print("GFV: ", GFV.shape, type(GFV))
            # print(GFV.cpu())
            # torch.save(PN_feature, "PN.pt")
            # torch.save(GFV, "GFV.pt")

            loss = criterion(pred, target.long(), trans_feat)
            all_train_loss.append(loss.item()) # Save training loss
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        record_save["train"]["instance_acc"].append(train_instance_acc)
        #record_save["train"]["expansion_loss"].append(expansion_loss.mean().item())
        record_save["train"]["expansion_loss"].append(0)
        record_save["train"]["cls_loss"].append(loss.item())
        #record_save["train"]["total_loss"].append(expansion_loss.mean().item()+loss.item())
        record_save["train"]["total_loss"].append(0)
    
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        # TEST MODE
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.module.eval(), testDataLoader)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + 'best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            if epoch % 20 == 0:
                torch.save(state, os.path.join(str(checkpoints_dir), "model_%04d.pth" %(epoch)))
                save_results(experiment_dir)
            global_epoch += 1

        # Save training and testing results

    logger.info('End of training...')
