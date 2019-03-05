import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import videotransforms
import numpy as np
from pytorch_i3d import InceptionI3d
from taiwan_dataset import Taiwan as Dataset

def run(max_steps=64e3,load_model='',root='/l/vision/v7/wang617/taiwan', batch_size=1, save_dir=''):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = Dataset(root,test_transforms, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)    
    i3d = InceptionI3d(400, in_channels=3)
    #i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode
    count = 0
    start = time.time()
    for data in dataloader:
            # get the inputs
        inputs, label, name = data
        label = str(label.numpy()[0])
        b,c,t,h,w = inputs.shape
        inputs = Variable(inputs.cuda(), volatile=True)
        features = i3d.extract_features(inputs)
        np.save(os.path.join(save_dir,name[0]),features.squeeze().data.cpu().numpy())
        f = open('/l/vision/v7/wang617/taiwan_data/i3d_feature_list.txt','a')
        f.writelines([name[0],',',label,'\n'])
        count = count +1
        if count%100 ==0:
            current = time.time()
            print('Count {:2},|' 'running time:{:.2f} sec'.format(count,current-start))
    f.close()
if __name__ == '__main__':
    # need to add argparse
    run(root=args.root, load_model=args.load_model, save_dir=args.save_dir)
