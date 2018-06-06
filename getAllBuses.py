import os
import numpy as np
from PIL import Image
import pathlib


def get_bus(image_box, bus_color):
#    image_box.show()
    color_dict = {'1': 'Green', '2': 'Yellow', '3': 'White', '4': 'Grey', '5': 'Blue', '6': 'Red'}
    busImageName = color_dict[str(bus_color)] + str(get_bus.counter[bus_color]) + ".jpg"
#    image_box_path = os.path.abspath(os.path.join(trainBusesFolder, color_dict[str(bus_color)]))
#    pathlib.Path(image_box_path).mkdir(parents=True, exist_ok=True) #Create Folder if not exist
#    image_box_path = os.path.abspath(os.path.join(image_box_path, busImageName))
    if(get_bus.counter[bus_color] % 5 == 0):
        image_box_path = os.path.abspath(os.path.join(trainBusesFolder, 'test'))
    else:
        image_box_path = os.path.abspath(os.path.join(trainBusesFolder, 'train'))
    pathlib.Path(image_box_path).mkdir(parents=True, exist_ok=True)
    image_box_path = os.path.abspath(os.path.join(image_box_path, busImageName))
    image_box.save(image_box_path)
    get_bus.counter[bus_color] += 1
    

get_bus.counter = [0, 1, 1, 1, 1, 1, 1] #first element is for easy reading


#for K-Means data
trainFolderPath = os.path.abspath(os.path.join(os.path.curdir,'train'))
trainImageFolder = os.path.abspath(os.path.join(trainFolderPath, 'Images'))
trainBusesFolder = os.path.abspath(os.path.join(trainFolderPath, 'Buses'))


annotationsOrigFileName = 'annotationsTrain.txt' #input annotations File Name
annotationsOrigFilePath = os.path.abspath(os.path.join(trainFolderPath, annotationsOrigFileName))
annotationsOrigFile = open(annotationsOrigFilePath,"r")

for line in annotationsOrigFile:    #Read File Line by Line
    #Get image
    imageName = line.split(":")[0] 
    imagePath = os.path.abspath(os.path.join(trainImageFolder, imageName))
    image=Image.open(imagePath)
#    image.show()
    imageObjects = line.split(":")[1]
        
    #Handle Objects Parameters
    listObjects=eval(imageObjects.split()[0])   #Split into list of lists
    
    if(type(listObjects) is tuple):             #Handle images with MORE than 1 bus
        for i in listObjects:  
            #xmin=i[0] ; ymin=i[1] ; xmax=xmin + i[2] ; ymax=ymin + i[3]
            im_box=image.crop((i[0], i[1], i[0] + i[2], i[1] + i[3]))
            color=i[4]
            get_bus(im_box, color)
    if(type(listObjects) is list):             #Handle images with 1 bus
        #xmin=listObjects[0] ; ymin=listObjects[1] ; xmax=xmin + listObjects[2] ; ymax=ymin + listObjects[3]
        im_box=image.crop((listObjects[0], listObjects[1], listObjects[0] + listObjects[2], listObjects[1] + listObjects[3]))
        color=listObjects[4]
        get_bus(im_box, color)
        
annotationsOrigFile.close()
