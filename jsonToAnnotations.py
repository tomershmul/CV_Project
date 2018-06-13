import os
import sys
import json
import numpy as np
from PIL import Image
from cropBuses import crop_one_bus
from busColorParameters import *
from busColorDetection import generate_dataset_of_one
from busColorPredict import predict

def json_to_ann  (yoloOutputPath, imagesPath, testAnnFilePath):
    print("Running json_to_ann function")
    testAnnFile = open(testAnnFilePath,"w")
    
    #Open json file for read
    for filename in os.listdir(yoloOutputPath):
        if filename[-5:] != ".json":
            continue
        jsonFileName = filename
        imageName = jsonFileName.split(".")[0]
        imageName += ".JPG"
        imagePath = os.path.abspath(os.path.join(imagesPath, imageName))
        
        jsonOutputFilePath = os.path.abspath(os.path.join(yoloOutputPath, jsonFileName))
        jsonOutputFile = open(jsonOutputFilePath,"r")
        
        data = json.load(jsonOutputFile)
        jsonOutputFile.close() # close json file
        
        #Image name
        strText = imageName + ":"
        testAnnFile.write(strText)
        
        for i in range(0,len(data)):
            ##### Crop bus
            image=Image.open(imagePath)
            im_box = image.crop((data[i]["topleft"]["x"],data[i]["topleft"]["y"],data[i]["bottomright"]["x"],data[i]["bottomright"]["y"]))
            # Maybe don't need to save to disk# crop_one_bus(im_box,yoloOutputPath,imageName,i)
            ##### Predict bus color
            image_set, image_label_onehot = generate_dataset_of_one(im_box, net_classes, size, Height, Width, color_dict_num)
            color_prediction = predict(image_set, image_label_onehot)
            
            #####
            if i != 0:
                testAnnFile.write(",")
            xmin = data[i]["topleft"]["x"]
            ymin = data[i]["topleft"]["y"]
            width = data[i]["bottomright"]["x"] - xmin
            height = data[i]["bottomright"]["y"] - ymin
            strText = "[" + str(xmin) + "," + str(ymin) + "," + str(width) + "," + str(height) + "," + str(color_prediction[0]+1) + "]" 
            testAnnFile.write(strText)
        
        testAnnFile.write("\n")
    
    testAnnFile.close()

def main():
    i = 0
    print("Running jsonToAnnotation script")
    #Open file for write
    yoloOutputPath = os.path.abspath(os.path.join(os.path.curdir,'my_img/out'))
    testAnnFileName = 'my_ann' 
    testAnnFilePath = os.path.abspath(os.path.join(yoloOutputPath, testAnnFileName))
    while(i < len(sys.argv)):
        arg = sys.argv[i]
        #print(i)
        #print("jsonToAnnotation arg" + str(i) + " " + arg)
        if('-myTestFolderPath' in arg):
            yoloOutputPath = sys.argv[i + 1]
            yoloOutputPath = os.path.join(os.getcwd(), yoloOutputPath)
        if('-myTestAnnFileName' in arg):
            testAnnFileName = sys.argv[i + 1]
            testAnnFilePath = os.path.join(os.getcwd(), testAnnFileName)
        i += 1
    
    json_to_ann(yoloOutputPath, testAnnFilePath)
        

if __name__ == "__main__":
    main()
    
    
