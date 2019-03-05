import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import numpy as np
import os
import os.path
import cv2
import random

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

def make_dataset(root):
    dataset = []
    for videoName in os.listdir(root):
        videoFramePath = os.path.join(root,videoName)
        for round in range(1,4):
            video_name = videoName
            idx = str(round).zfill(6)
            clip_start = 1
            clip_end = len(os.listdir(videoFramePath))
            index = random.randint(clip_start,clip_end)
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
            dataset.append((video_name,indexList,index))
            print(indexList)
    return dataset

class Hevi(data_utl.Dataset):

    def __init__(self,root,transforms=None, save_dir='', num=0):
        
        self.data = make_dataset(root)
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
        video_name,indexList,idx = self.data[index]
        #if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
        #    return 0, 0, vid
        imgs = load_rgb_frames(self.root,video_name,indexList)
        imgs = self.transforms(imgs)
        idx = str(idx).zfill(6)
        return video_to_tensor(imgs),video_name+'_'+idx

    def __len__(self):
        return len(self.data)
