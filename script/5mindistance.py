import numpy as np
import pickle
import os
import os.path
import time
featurePath = '/l/vision/v7/wang617/HEV-I/i3d_random_feature'
ananFeaturePath = '/l/vision/v7/wang617/anan_dataset/Volumes/GRACE/anan/i3d_feature'
pklFilePath = '/l/vision/v7/wang617/anan_dataset/Volumes/GRACE/anan/i3d_5mindistance_dic.pkl'
featureMatrix = np.empty([1024,0])
anan = {}
for heviFeature in os.listdir(featurePath):
    loadPath = os.path.join(featurePath,heviFeature)
    feature = np.load(loadPath)
    feature = feature.reshape(-1,1)
    featureMatrix = np.append(featureMatrix,feature,axis=1)
featureMatrix = featureMatrix.T
count = 0
start = time.time()
with open('/l/vision/v7/wang617/anan_dataset/Volumes/GRACE/anan/i3d_feature_list.txt', 'r') as ananFile:
    ananFeatureList = ananFile.readlines()
    for ananFeatureName in ananFeatureList:
        sourceData = ananFeatureName.split(',')
        ananFeaFilName = sourceData[0]
        label = sourceData[1][0]
        ananI3dFeature = np.load(os.path.join(ananFeaturePath,ananFeaFilName+'.npy'))
        distanceList = []
        for heviFeature in featureMatrix:
            distance = np.linalg.norm(ananI3dFeature - heviFeature)
            distanceList.append(distance)
        distanceList.sort()
        distanceList = distanceList[0:5]
        framID = ananFeaFilName[-6:]
        clip_index = ananFeaFilName[-13:-7]
        videoName = ananFeaFilName[0:(len(sourceData[0])-14)]
        anan[ananFeaFilName] = {
        'video_name':videoName,
        'clip_index':clip_index,
        'frameID':framID,
        'distance': distanceList,
        'target': label,
        }
        count = count +1
        if count%1000 ==0:
            current = time.time()
            print('Count {:2},|' 'running time:{:.2f} sec'.format(count,current-start))
ananFile.close()
with open(pklFilePath, 'wb') as pk:
    pickle.dump(anan, pk)
    print('Write pickle file to {}'.format(pklFilePath))
pk.close()