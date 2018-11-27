from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from imageai.Detection import ObjectDetection
import os
from operator import itemgetter
import csv
execution_path2 = '/media/zhaoer/SSD/CV project/deploy/trainval/0cec3d1f-544c-4146-8632-84b1f9fe89d3'
# image_path = execution_path2 +'/*_image.jpg'
# images = glob(image_path)
files = glob('/media/zhaoer/SSD/CV project/deploy/test/*')
# files = glob('/media/zhaoer/SSD/CV project/deploy/trainval/1dac619e-4123-42e9-bf11-8df2db3facf7/*')

# 找出test下面所有的文件夹
def get_directories(files):
	dirs = [f 
		for f in files
			if os.path.isdir(f)]
	return dirs
folders = get_directories(files)

execution_path = os.getcwd()
output_array = []
no_car_images = []
counter=0
counter2=0
print (counter)
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

for i in folders:

	# if (counter >3):
	# 	break
	# [41:82]
	# [45:86]
	image_path = i +'/*_image.jpg'
	images = sorted(glob(image_path))
	for image in images:
		# if (counter >3):
		# 	break
		# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path2 , "0000_image.jpg"), output_image_path=os.path.join(execution_path , "image2new.jpg"), minimum_percentage_probability=30)
		# if not(counter2<0):
		detections = detector.detectObjectsFromImage(input_image=image, output_image_path=os.path.join(execution_path , "image2new.jpg"))
		if len(detections):
			# sortedlist = sorted(detections, key=itemgetter('percentage_probability'), reverse=True)
			# for i,dic in enumerate(sortedlist):
			#         # if dic["name"] == 'car':
		 #            a = [image[41:82],str(dic["box_points"][0]),str(dic["box_points"][1]),str(dic["box_points"][2]),str(dic["box_points"][3]),dic["name"] , str(dic["percentage_probability"])]
		 #            print (a)
		 #            output_array.append(a)
		    break
		            # break
		else:
			print ("car object absent")
			a = [image[41:82]]
			print (a)
			no_car_images.extend(a)
				
		# counter2 +=1
# print ("car object absent")
# print (output_array)
with open("classification_output.csv",'w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(output_array)

print (no_car_images)
with open("no_car_images.csv",'w') as resultFile:
	wr = csv.writer(resultFile, dialect='excel')
	wr.writerows(no_car_images)
