import os
import numpy as np
from PIL import Image
import pathlib
import re
import Augmentor
from busColorParameters import *
import shutil


def get_bus(image_box, bus_color, imageName, test_flag):
#    image_box.show()
    color_dict = {'1': 'Green', '2': 'Yellow', '3': 'White', '4': 'Grey', '5': 'Blue', '6': 'Red'}
    busImageName = os.path.basename(str(imageName)).split('.')[0] + '_' + color_dict[str(bus_color)] + str(get_bus.counter[bus_color]) + ".jpg"
#    image_box_path = os.path.abspath(os.path.join(trainBusesFolder, color_dict[str(bus_color)]))
#    pathlib.Path(image_box_path).mkdir(parents=True, exist_ok=True) #Create Folder if not exist
#    image_box_path = os.path.abspath(os.path.join(image_box_path, busImageName))

#    if(get_bus.counter[bus_color] % 5 == 0):
    if(test_flag==1):
        image_box_path = os.path.abspath(os.path.join(cropedBusesFolderPath, 'test'))
    else:
        image_box_path = os.path.abspath(os.path.join(cropedBusesFolderPath, 'train'))

#    image_box_path = os.path.abspath(os.path.join(trainBusesFolder, 'train'))
    pathlib.Path(image_box_path).mkdir(parents=True, exist_ok=True)
    image_box_path = os.path.abspath(os.path.join(image_box_path, busImageName))
    image_box.save(image_box_path)
    get_bus.counter[bus_color] += 1
    
#Remove folders from Previous run
if os.path.exists(trainCropedBusesFolder):    
    shutil.rmtree(trainCropedBusesFolder, ignore_errors=False, onerror=None)
if os.path.exists(testCropedBusesFolder):    
    shutil.rmtree(testCropedBusesFolder, ignore_errors=False, onerror=None)

get_bus.counter = [0, 1, 1, 1, 1, 1, 1] #first element is for easy reading

annotationsOrigFile = open(annotationsOrigFilePath,"r")

for line in annotationsOrigFile:    #Read File Line by Line
    #Get image
    imageName = line.split(":")[0]
    if re.match('DSCF106', imageName):
        test_flag=1
    else:
        test_flag=0
    imagePath = os.path.abspath(os.path.join(allImagesFolder, imageName))
    image=Image.open(imagePath)
#    image.show()
    imageObjects = line.split(":")[1]
    imageSize = image.size
    #Handle Objects Parameters
    listObjects=eval(imageObjects.split()[0])   #Split into list of lists
    
    crop_factor=0.05
    
    if(type(listObjects) is tuple):             #Handle images with MORE than 1 bus
        for i in listObjects:  
            if test_flag:                       #do not change crop size for test images
                im_box=image.crop((i[0], i[1], i[0] + i[2], i[1] + i[3]))
            else:
                crop_delta_x=crop_factor*i[2]
                crop_delta_y=crop_factor*i[3]
                xmin=max(0,i[0] - crop_delta_x)
                ymin=max(0,i[1] - crop_delta_y)
                xmax=min(imageSize[0], (i[0] + i[2])+crop_delta_x)
                ymax=min(imageSize[1], (i[1] + i[3])+crop_delta_y)
                im_box=image.crop((xmin, ymin, xmax, ymax))
            #xmin=i[0] ; ymin=i[1] ; xmax=xmin + i[2] ; ymax=ymin + i[3]
#            im_box=image.crop((i[0], i[1], i[0] + i[2], i[1] + i[3]))
#            im_box=image.crop((xmin, ymin, xmax, ymax))
            color=i[4]
            get_bus(im_box, color, imageName, test_flag)
    if(type(listObjects) is list):             #Handle images with 1 bus
        if test_flag:                          #do not change crop size for test images
            im_box=image.crop((listObjects[0], listObjects[1], listObjects[0] + listObjects[2], listObjects[1] + listObjects[3]))
        else:
            crop_delta_x=crop_factor*listObjects[2]
            crop_delta_y=crop_factor*listObjects[3]
            xmin=max(0,listObjects[0] - crop_delta_x)
            ymin=max(0,listObjects[1] - crop_delta_y)
            xmax=min(imageSize[0], (listObjects[0] + listObjects[2])+crop_delta_x)
            ymax=min(imageSize[1], (listObjects[1] + listObjects[3])+crop_delta_y)
            im_box=image.crop((xmin, ymin, xmax, ymax))
        #xmin=listObjects[0] ; ymin=listObjects[1] ; xmax=xmin + listObjects[2] ; ymax=ymin + listObjects[3]
#        im_box=image.crop((listObjects[0], listObjects[1], listObjects[0] + listObjects[2], listObjects[1] + listObjects[3]))
#        im_box=image.crop((xmin, ymin, xmax, ymax))
        color=listObjects[4]
        get_bus(im_box, color, imageName, test_flag)
               
annotationsOrigFile.close()

#image Augmentation

num_of_Aug_images = 100

p = Augmentor.Pipeline(trainCropedBusesFolder)
p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.crop_random(probability=1, percentage_area=0.8)
p.sample(num_of_Aug_images)

augImagesFolder = os.path.abspath(os.path.join(trainCropedBusesFolder, 'output'))
augImagesFolder = os.path.join(augImagesFolder, '')

aug_idx=0
for aug_img in os.listdir(augImagesFolder):
    print(aug_img)
    aug_img_name = re.search('train_original_(.*)\.jpg_', aug_img, re.IGNORECASE).group(1)
    aug_img_name = aug_img_name + '_augmentated_' + str(aug_idx)
    print(aug_img_name)
    os.rename(augImagesFolder + str(aug_img), trainCropedBusesFolder + str(aug_img_name) + ".jpg")
    aug_idx += 1

if os.path.exists(augImagesFolder):    
    shutil.rmtree(augImagesFolder, ignore_errors=False, onerror=None)