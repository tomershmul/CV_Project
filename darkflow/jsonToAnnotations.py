import os
import json
import numpy as np

#Open file for write
#testFolderPath = os.path.abspath(os.path.join(os.path.curdir,'my_img/out'))
#testAnnFileName = 'my_ann' 
testFolderPath = os.path.abspath(os.path.join(os.path.curdir,'train/Images/out'))
testAnnFileName = 'my_ann' 
testAnnFilePath = os.path.abspath(os.path.join(testFolderPath, testAnnFileName))
testAnnFile = open(testAnnFilePath,"w")


#Open json file for read
#jsonFileName = 'DSCF1016.json'
#imageName = jsonFileName.split(".")[0]
#imageName += ".JPG"

for filename in os.listdir(testFolderPath):
    if filename[-5:] != ".json":
        continue
    jsonFileName = filename
    imageName = jsonFileName.split(".")[0]
    imageName += ".JPG"
    
    jsonOutputFilePath = os.path.abspath(os.path.join(testFolderPath, jsonFileName))
    jsonOutputFile = open(jsonOutputFilePath,"r")
    
    data = json.load(jsonOutputFile)
    
    #Image name
    strText = imageName + ":"
    testAnnFile.write(strText)
    
    for i in range(0,len(data)):
        if i != 0:
            testAnnFile.write(",")
        xmin = data[i]["topleft"]["x"]
        ymin = data[i]["topleft"]["y"]
        width = data[i]["bottomright"]["x"] - xmin
        height = data[i]["bottomright"]["y"] - ymin
        strText = "[" + str(xmin) + "," + str(ymin) + "," + str(width) + "," + str(height) + ",1]" #TODO ('1')
        testAnnFile.write(strText)
    
    testAnnFile.write("\n")


testAnnFile.close()

    
