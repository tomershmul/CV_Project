import os
import numpy as np
from PIL import Image
import pathlib
import re
import Augmentor
from busColorParameters import *
import shutil
import random
from shutil import copy


def get_bus(image_box, bus_color, imageName, test_flag, aug_flag):
#    color_dict = {'1': 'Green', '2': 'Yellow', '3': 'White', '4': 'Grey', '5': 'Blue', '6': 'Red'}
    busImageName = os.path.basename(str(imageName)).split('.')[0] + '_' + color_dict[str(bus_color)] + str(get_bus.counter[bus_color]) + ".jpg"

    if(test_flag==1):
        image_box_path = testCropedBusesFolder
    elif (aug_flag==0):
        image_box_path = trainCropedBusesFolder
    else:
        image_box_path = augImagesFolder

    pathlib.Path(image_box_path).mkdir(parents=True, exist_ok=True)
    image_box_path = os.path.abspath(os.path.join(image_box_path, busImageName))
    image_box.save(image_box_path)
    get_bus.counter[bus_color] += 1
    
#YOLO train and test Folders
testYoloFolder = os.path.abspath(os.path.join(rootDir, 'test', 'Images'))
testYoloAnnotationsFolder = os.path.abspath(os.path.join(rootDir, 'test', 'Annotations'))
trainYoloFolder = os.path.abspath(os.path.join(rootDir, 'train', 'Images'))
trainYoloAnnotationsFolder = os.path.abspath(os.path.join(rootDir, 'train', 'Annotations'))

#Remove folders from previous run
if os.path.exists(testYoloFolder):    
    shutil.rmtree(testYoloFolder, ignore_errors=False, onerror=None)
if os.path.exists(testYoloAnnotationsFolder):    
    shutil.rmtree(testYoloAnnotationsFolder, ignore_errors=False, onerror=None)
if os.path.exists(trainYoloFolder):    
    shutil.rmtree(trainYoloFolder, ignore_errors=False, onerror=None)
if os.path.exists(trainYoloAnnotationsFolder):    
    shutil.rmtree(trainYoloAnnotationsFolder, ignore_errors=False, onerror=None)

if os.path.exists(trainCropedBusesFolder):    
    shutil.rmtree(trainCropedBusesFolder, ignore_errors=False, onerror=None)
if os.path.exists(testCropedBusesFolder):    
    shutil.rmtree(testCropedBusesFolder, ignore_errors=False, onerror=None)
    
#Make Dir for YOLO train and test
pathlib.Path(testYoloFolder).mkdir(parents=True, exist_ok=True)
pathlib.Path(testYoloAnnotationsFolder).mkdir(parents=True, exist_ok=True)
pathlib.Path(trainYoloFolder).mkdir(parents=True, exist_ok=True)
pathlib.Path(trainYoloAnnotationsFolder).mkdir(parents=True, exist_ok=True)

get_bus.counter = [0, 1, 1, 1, 1, 1, 1] #first element is for easy reading

annotationsOrigFile = open(annotationsOrigFilePath,"r")

#Choose Random Images for Test Set
num_of_Images=sum(1 for _ in annotationsOrigFile)
num_of_TestImages=int(0.1*num_of_Images)
testLines = random.sample(range(num_of_Images),num_of_TestImages)
annotationsOrigFile.close()

annotationsOrigFile = open(annotationsOrigFilePath,"r")
line_idx=0

for line in annotationsOrigFile:    #Read File Line by Line
    #Get image
    imageName = line.split(":")[0]
    imagePath = os.path.abspath(os.path.join(allImagesFolder, imageName))
    annotationPath = os.path.abspath(os.path.join(allAnnotationsFolder, imageName.split('.')[0]) + '.xml')
#    if re.match('DSCF106', imageName):
    if line_idx in testLines:
        test_flag=1
        copy(imagePath, testYoloFolder)
        copy(annotationPath, testYoloAnnotationsFolder)
    else:
        test_flag=0
        copy(imagePath, trainYoloFolder)
        copy(annotationPath, trainYoloAnnotationsFolder)
    image=Image.open(imagePath)
#    image.show()
    imageObjects = line.split(":")[1]
    imageSize = image.size
    #Handle Objects Parameters
    listObjects=eval(imageObjects.split()[0])   #Split into list of lists
    
#    crop_factor=0.01
    
    if(type(listObjects) is tuple):             #Handle images with MORE than 1 bus
        for i in listObjects:  
            color=i[4]
            if test_flag:                       #do not change crop size for test images
                aug_flag=0
                im_box=image.crop((i[0], i[1], i[0] + i[2], i[1] + i[3]))
                get_bus(im_box, color, imageName, test_flag, aug_flag)
            else:
                aug_flag=1
                crop_delta_x=crop_factor*i[2]
                crop_delta_y=crop_factor*i[3]
                xmin=max(0,i[0] - crop_delta_x)
                ymin=max(0,i[1] - crop_delta_y)
                xmax=min(imageSize[0], (i[0] + i[2])+crop_delta_x)
                ymax=min(imageSize[1], (i[1] + i[3])+crop_delta_y)
                im_box=image.crop((xmin, ymin, xmax, ymax))
                get_bus(im_box, color, imageName, test_flag, aug_flag)
                
                aug_flag=0
                im_box=image.crop((i[0], i[1], i[0] + i[2], i[1] + i[3]))
                #xmin=i[0] ; ymin=i[1] ; xmax=xmin + i[2] ; ymax=ymin + i[3]
                get_bus(im_box, color, imageName, test_flag, aug_flag)
            
    if(type(listObjects) is list):             #Handle images with 1 bus
        color=listObjects[4]
        if test_flag:                          #do not change crop size for test images
            aug_flag=0
            im_box=image.crop((listObjects[0], listObjects[1], listObjects[0] + listObjects[2], listObjects[1] + listObjects[3]))
            get_bus(im_box, color, imageName, test_flag, aug_flag)
        else:
            aug_flag=1
            crop_delta_x=crop_factor*listObjects[2]
            crop_delta_y=crop_factor*listObjects[3]
            xmin=max(0,listObjects[0] - crop_delta_x)
            ymin=max(0,listObjects[1] - crop_delta_y)
            xmax=min(imageSize[0], (listObjects[0] + listObjects[2])+crop_delta_x)
            ymax=min(imageSize[1], (listObjects[1] + listObjects[3])+crop_delta_y)
            im_box=image.crop((xmin, ymin, xmax, ymax))
            get_bus(im_box, color, imageName, test_flag, aug_flag)
            
            aug_flag=0
            im_box=image.crop((listObjects[0], listObjects[1], listObjects[0] + listObjects[2], listObjects[1] + listObjects[3]))
            #xmin=listObjects[0] ; ymin=listObjects[1] ; xmax=xmin + listObjects[2] ; ymax=ymin + listObjects[3]
            get_bus(im_box, color, imageName, test_flag, aug_flag)
            
    line_idx+=1
    
annotationsOrigFile.close()

#image Augmentation

#num_of_Aug_images = 800

p = Augmentor.Pipeline(augImagesFolder)
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.crop_random(probability=1, percentage_area=0.85)
p.random_brightness(probability=0.6, min_factor=0.5, max_factor=1.5)
p.sample(num_of_Aug_images)



augmentorImagesFolder = os.path.abspath(os.path.join(augImagesFolder, 'output'))
augmentorImagesFolder = os.path.join(augmentorImagesFolder, '')

aug_idx=0
for aug_img in os.listdir(augmentorImagesFolder):
#    print(aug_img)
    aug_img_name = re.search('aug_img_original_(.*)\.jpg_', aug_img, re.IGNORECASE).group(1)
    aug_img_name = aug_img_name + '_augmentated_' + str(aug_idx)
#    print(aug_img_name)
    os.rename(augmentorImagesFolder + str(aug_img), trainCropedBusesFolder + str(aug_img_name) + ".jpg")
    aug_idx += 1

if os.path.exists(augImagesFolder):    
    shutil.rmtree(augImagesFolder, ignore_errors=False, onerror=None)