import os
from PIL import Image
import numpy as np

#Gets bounding box parameters and file path
def bndbox_parameters(xmin, ymin, xmax, ymax, file):
    file.write("\t\t<bndbox>\n")
            
    strText = "\t\t\t<xmin>" + str(xmin) + "</xmin>\n"
    file.write(strText)
    strText = "\t\t\t<ymin>" + str(ymin) + "</ymin>\n"
    file.write(strText)   
    strText = "\t\t\t<xmax>" + str(xmax) + "</xmax>\n"
    file.write(strText)
    strText = "\t\t\t<ymax>" + str(ymax) + "</ymax>\n"
    file.write(strText)
            
    file.write("\t\t</bndbox>\n") #Finish Bounding Box

#Gets image path & file path, print the size section
def image_parameters(imagePath, file):
    width, height, depth = np.array(Image.open(imagePath)).shape
   
#    print (width, height, depth)
    
    strText = "\t<size>\n"
    annotationsNewFile.write(strText)
    strText = "\t\t<width>" + str(width) + "</width>\n"
    annotationsNewFile.write(strText)
    strText = "\t\t<height>" + str(height) + "</height>\n"
    annotationsNewFile.write(strText)
    strText = "\t\t<depth>" + str(depth) + "</depth>\n"
    annotationsNewFile.write(strText)
    strText = "\t</size>\n"
    annotationsNewFile.write(strText)
    
    
#Files Path and Names    
trainFolderPath = os.path.abspath(os.path.join(os.path.curdir,'train'))

annotationsNewFilesFolder = os.path.abspath(os.path.join(trainFolderPath, 'Annotations'))
trainImageFolder = os.path.abspath(os.path.join(trainFolderPath, 'Images'))

annotationsOrigFileName = 'annotationsTrain.txt' #input annotations File Name
annotationsOrigFilePath = os.path.abspath(os.path.join(trainFolderPath, annotationsOrigFileName))

#Open file
annotationsOrigFile = open(annotationsOrigFilePath,"r")

for line in annotationsOrigFile:    #Read File Line by Line
    #Get annotaions file name and image name
    imageName = line.split(":")[0]
    fileName = imageName.split(".")[0]
    fileName += ".xml"
    
    imageObjects = line.split(":")[1]
    
    #Get File Path
    annotationsNewFilePath = os.path.abspath(os.path.join(annotationsNewFilesFolder,fileName))
    annotationsNewFile = open(annotationsNewFilePath,"w")
    
    annotationsNewFile.write("<annotation>\n")
    
    #Handle Image & File Names
    strText = "\t<filename>" + imageName + "</filename>\n"
    annotationsNewFile.write(strText)
    
    #Handle Image Size Parameters
    #imagePath = os.path.abspath(os.path.join(trainImageFolder,imageName))
    image_parameters(os.path.abspath(os.path.join(trainImageFolder,imageName)),annotationsNewFile)
    
    #Handle Objects Parameters
    listObjects=eval(imageObjects.split()[0])   #Split into list of lists
    
    if(type(listObjects) is tuple):             #Handle images with MORE than 1 bus
        for i in listObjects:  
            annotationsNewFile.write("\t<object>\n")
            strText = "\t\t<name>bus</name>\n"
            annotationsNewFile.write(strText)
            
            #xmin=i[0] ; ymin=i[1] ; xmax=xmin + i[2] ; ymax=ymin + i[3]
            bndbox_parameters(i[0], i[1], i[0] + i[2], i[1] + i[3], annotationsNewFile)
            annotationsNewFile.write("\t</object>\n")   #Finish Object Box
    
    if(type(listObjects) is list):             #Handle images with 1 bus
        annotationsNewFile.write("\t<object>\n")
        strText = "\t\t<name>bus</name>\n"
        annotationsNewFile.write(strText)
        
        #xmin=listObjects[0] ; ymin=listObjects[1] ; xmax=xmin + listObjects[2] ; ymax=ymin + listObjects[3]
        bndbox_parameters(listObjects[0], listObjects[1], listObjects[0] + listObjects[2], listObjects[1] + listObjects[3], annotationsNewFile)
        annotationsNewFile.write("\t</object>\n")   #Finish Object Box
        
    annotationsNewFile.write("</annotation>\n")    
    annotationsNewFile.close()

annotationsOrigFile.close()
