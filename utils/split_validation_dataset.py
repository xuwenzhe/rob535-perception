import pandas as pd
import numpy as np
import os
import shutil


df = pd.read_csv('./data-old/train_labels.csv')

train, validate = np.split(df.sample(frac=1), [int(0.9*len(df))])

train.to_csv('./data/train_labels.csv',index=False)
validate.to_csv('./data/test_labels.csv',index=False)


path = '/home/wenzhe/Desktop/rob535-task1'
for row in range(len(train)):
	imgname = train.iloc[row]['filename']
	shutil.copy(path+'/images-old/train/'+imgname, path+'/images/train')

for row in range(len(validate)):
	imgname = validate.iloc[row]['filename']
	shutil.copy(path+'/images-old/train/'+imgname, path+'/images/test')