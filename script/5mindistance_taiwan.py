import numpy as np
import pickle
import os
import os.path
import time
featurePath = '/l/vision/v7/wang617/HEV-I/i3d_random_feature'
taiwanFeaturePath = '/l/vision/v7/wang617/taiwan_data/i3d_feature'
pklFilePath = '/l/vision/v7/wang617/taiwan_data/i3d_5mindistance_dic.pkl'
featureMatrix = np.empty([1024,0])
taiwan = {}
for heviFeature in os.listdir(featurePath):
    loadPath = os.path.join(featurePath,heviFeature)
    feature = np.load(loadPath)
    feature = feature.reshape(-1,1)
    featureMatrix = np.append(featureMatrix,feature,axis=1)
featureMatrix = featureMatrix.T
count = 0
start = time.time()
with open('/l/vision/v7/wang617/taiwan_data/i3d_feature_list.txt', 'r') as taiwanFile:
    taiwanFeatureList = taiwanFile.readlines()
    for taiwanFeatureName in taiwanFeatureList:
        sourceData = taiwanFeatureName.split(',')
        taiwanFeaFilName = sourceData[0]
        label = sourceData[1][0]
        taiwanI3dFeature = np.load(os.path.join(taiwanFeaturePath,taiwanFeaFilName+'.npy'))
        distanceList = []
        for heviFeature in featureMatrix:
            distance = np.linalg.norm(taiwanI3dFeature - heviFeature)
            distanceList.append(distance)
        distanceList.sort()
        distanceList = distanceList[0:5]
        framID = taiwanFeaFilName[-6:]
        videoName = taiwanFeaFilName[0:6]
        taiwan[taiwanFeaFilName] = {
        'video_name':videoName,
        'frameID':framID,
        'distance': distanceList,
        'target': label,
        }
        count = count +1
        if count%1000 ==0:
            current = time.time()
            print('Count {:2},|' 'running time:{:.2f} sec'.format(count,current-start))
taiwanFile.close()
with open(pklFilePath, 'wb') as pk:
    pickle.dump(taiwan, pk)
    print('Write pickle file to {}'.format(pklFilePath))
pk.close()
