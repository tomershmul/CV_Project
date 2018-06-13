import os
import numpy as np
from PIL import Image
import pathlib


def crop_one_bus(image_box,testImageFolder,imageName,index):
    busImageName = imageName + str(index) + ".jpg"
    image_box_path = os.path.abspath(os.path.join(testImageFolder, 'croped_buses'))
    pathlib.Path(image_box_path).mkdir(parents=True, exist_ok=True)
    image_box_path = os.path.abspath(os.path.join(image_box_path, busImageName))
    image_box.save(image_box_path)


def crop_buses (annotationsFilePath, testImageFolder):
    print("Running crop_buses function")
    #testImageFolder = os.path.abspath(os.path.join(testFolderPath, 'Images'))
    annotationsFile = open(annotationsFilePath,"r")
    for line in annotationsFile:    #Read File Line by Line
        bus_counter = 1
        #Get image
        imageName = line.split(":")[0] 
        imagePath = os.path.abspath(os.path.join(testImageFolder, imageName))
        image=Image.open(imagePath)
        imageObjects = line.split(":")[1]
            
        #Handle Objects Parameters
        listObjects=eval(imageObjects.split()[0])   #Split into list of lists
        
        if(type(listObjects) is tuple):             #Handle images with MORE than 1 bus
            for i in listObjects:  
                #xmin=i[0] ; ymin=i[1] ; xmax=xmin + i[2] ; ymax=ymin + i[3]
                im_box=image.crop((i[0], i[1], i[0] + i[2], i[1] + i[3]))
                crop_one_bus(im_box,testImageFolder,imageName,bus_counter)
                bus_counter += 1

        if(type(listObjects) is list):             #Handle images with 1 bus
            #xmin=listObjects[0] ; ymin=listObjects[1] ; xmax=xmin + listObjects[2] ; ymax=ymin + listObjects[3]
            im_box=image.crop((listObjects[0], listObjects[1], listObjects[0] + listObjects[2], listObjects[1] + listObjects[3]))
            crop_one_bus(im_box,testImageFolder,imageName,bus_counter)
            bus_counter += 1
    
    annotationsFile.close()

def main():
    print("Running cropBuses script")
    testFolderPath = os.path.abspath(os.path.join(os.path.curdir,'test'))
    testImageFolder = os.path.abspath(os.path.join(testFolderPath, 'Images'))
    
    
    annotationsFileName = 'annotationsTrain.txt' #input annotations File Name
    annotationsFilePath = os.path.abspath(os.path.join(testFolderPath, annotationsFileName))
    
    crop_buses (annotationsFilePath, testImageFolder)
        

if __name__ == "__main__":
    main()
    
    

