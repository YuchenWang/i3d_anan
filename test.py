import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import numpy as np
import os
import os.path
import cv2
import pickle

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def load_rgb_frames(image_dir, video_name, indexList):
  frames = []
  for i in indexList:
    img = cv2.imread(os.path.join(image_dir, video_name, str(i).zfill(6)+'.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (226,226), interpolation=cv2.INTER_AREA)
    w,h,c = img.shape
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def make_dataset(pkpath,root,num_classes=157):
    dataset = []
    with open(pkpath, 'rb') as dic:
        datasetDic = pickle.load(dic)
    del datasetDic['_000000']
    for clip in datasetDic.keys():
        video_name = datasetDic[clip]['video_name']
        idx = datasetDic[clip]['idx']
        clip_start = int(datasetDic[clip]['clip_start'])
        clip_end = int(datasetDic[clip]['clip_end'])
        target = datasetDic[clip]['target']
        for index in range(clip_start,clip_end+1):
            if index-clip_start >= 7:
                if clip_end - index >= 8:
                    indexList = list(range(index-7,index+9))
                else:
                    indexList = list(range(index-7,clip_end+1))
                    addLength = 16-len(indexList)
                    tmp = [clip_end for n in range(addLength)]
                    indexList = indexList+tmp
            else:
                indexList = list(range(clip_start,index+9))
                addLength = 16-len(indexList)
                tmp = [clip_start for n in range(addLength)]
                indexList = tmp+indexList
            label = target[index - clip_start]
            dataset.append((video_name,idx,indexList,label,index))
    return dataset
class Anan(data_utl.Dataset):

    def __init__(self,pkpath,root,transforms=None, save_dir='', num=0):

        self.data = make_dataset(pkpath,root)
        self.transforms = transforms
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        video_name,idx,indexList,label,index = self.data[index]
        #if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
        #    return 0, 0, vid
        print(video_name,idx,indexList)

        imgs = load_rgb_frames(self.root,video_name,indexList)
        #imgs = self.transforms(imgs)
        index = str(index).zfill(6)
        return video_to_tensor(imgs),label, video_name+'_'+idx+'_'+index

    def __len__(self):
        return len(self.data)

pkpath='/l/vision/v7/wang617/anan_dataset/Volumes/GRACE/anan/AnAn_Dataset_Labels.pkl'
root='/l/vision/v7/wang617/anan_dataset/Volumes/GRACE/anan/frame'
dataset = Anan(pkpath,root, save_dir='/l/vision/v7/wang617/anan_dataset/Volumes/GRACE/anan/')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
for data in dataloader:
	print(data[2]) 
