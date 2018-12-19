import os
import shutil

path = '/home/wenzhe/Desktop/rob535-task1/all/deploy'



# train
# folders = os.listdir(path+'/trainval')

# for folder in folders:
# 	if len(folder) == 36:
# 		files = os.listdir(path+'/trainval/'+folder)
# 		for file in files:
# 			if file.endswith('jpg'):
# 				shutil.copy(path+'/trainval/'+folder+'/'+file, path+'/images/train')
# 				os.rename(path+'/images/train/'+file, path+'/images/train/'+folder+'-'+file)

folders = os.listdir(path+'/test')

for folder in folders:
	if len(folder) == 36:
		files = os.listdir(path+'/test/'+folder)
		for file in files:
			if file.endswith('jpg'):
				shutil.copy(path+'/test/'+folder+'/'+file, path+'/images/test')
				os.rename(path+'/images/test/'+file, path+'/images/test/'+folder+'-'+file)