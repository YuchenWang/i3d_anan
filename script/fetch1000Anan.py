import pandas as pd

loadcsvPath = "/Users/wangyuchen/Downloads/AnAn_Dataset_Labels.csv"
loadtxtPath = "/Users/wangyuchen/Downloads/A3D_final_videos.txt"
savecsvPath = "/Users/wangyuchen/Downloads/AnAn_Dataset_Labels_1000.csv"

data = pd.read_csv(loadcsvPath)
video_name = []
clip_index = []
clip_start = []
anomaly_start = []
anomaly_end = []
clip_end =[]
ego_envolve_or_not = []
ego_only = []
description = []
videoDic_1000 = {}

with open(loadtxtPath) as file:
	video = file.readlines()
for videoIdx in video:
	videoName = videoIdx[0:(len(videoIdx)-8)]
	clipIdx = videoIdx[-7:-1]
	if videoName in videoDic_1000:
		tmp = videoDic_1000[videoName]
		tmp.append(clipIdx)
		videoDic_1000[videoName] = tmp
	else:
		videoDic_1000[videoName] = [clipIdx]
print(videoDic_1000)
for i in range(0,1500):
	if data['video'][i] in videoDic_1000.keys():
		vName = data['video'][i]
		cIndex = str(data['clip_index'][i]).zfill(6)
		if cIndex in videoDic_1000[vName]:
			video_name.append(data['video'][i])
			clip_index.append(data['clip_index'][i])
			clip_start.append(data['clip_start'][i])
			anomaly_start.append(data['anomaly_start'][i])
			anomaly_end.append(data['anomaly_end'][i])
			clip_end.append(data['clip_end'][i])
			ego_envolve_or_not.append(data['ego_envolve_or_not'][i])
			ego_only.append(data['ego_only'][i])
			description.append(data['description'][i])
dataframe = pd.DataFrame({'video':video_name,'clip_index':clip_index,'clip_start':clip_start,'anomaly_start':anomaly_start,'anomaly_end':anomaly_end,'clip_end':clip_end,'ego_envolve_or_not':ego_envolve_or_not,'ego_only':ego_only,'description':description})
dataframe.to_csv(savecsvPath,index=False,sep=',')