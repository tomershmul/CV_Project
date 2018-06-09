import os

#    global color_dict, color_dict_num, trainFolderPath, trainFolderPath, modelPath, modelName, trainFolder, testFolder, trainFolder, testFolder, out_L1, out_L2, out_L3, out_L4, net_classes, num_of_itr, Width, Height, size, metaFile 

color_dict = {'1': 'Green', '2': 'Yellow', '3': 'White', '4': 'Grey', '5': 'Blue', '6': 'Red'}
color_dict_num = {'Green': 1, 'Yellow': 2, 'White': 3, 'Grey': 4, 'Blue': 5, 'Red': 6}

#CNN Parameters
out_L1=6
out_L2=6
out_L3=6
out_L4=50 #Fully Connected layer

net_classes=6
num_of_itr=1000


#Resize Parameters
Width = 224
Height = 224
size = Width, Height

rootDir = os.path.abspath(os.path.join(os.path.curdir))

annotationsOrigFileName = 'annotationsTrain.txt' #input annotations File Name
annotationsOrigFilePath = os.path.abspath(os.path.join(rootDir, annotationsOrigFileName))

allImagesFolder = os.path.abspath(os.path.join(rootDir, 'Images'))
cropedBusesFolderPath = os.path.abspath(os.path.join(rootDir,'Croped_buses'))
#trainFolderPath = os.path.abspath(os.path.join(trainFolderPath, 'Croped_buses'))
modelPath = os.path.abspath(os.path.join(cropedBusesFolderPath, 'Model'))
modelName = os.path.abspath(os.path.join(modelPath, 'busColorDetector'))

metaFile = modelName +'-'+str(num_of_itr)+'.meta'

trainCropedBusesFolder = os.path.abspath(os.path.join(cropedBusesFolderPath, 'train'))
trainCropedBusesFolder = os.path.join(trainCropedBusesFolder, '')

testCropedBusesFolder = os.path.abspath(os.path.join(cropedBusesFolderPath, 'test'))
testCropedBusesFolder = os.path.join(testCropedBusesFolder, '')

augImagesFolder = os.path.abspath(os.path.join(trainCropedBusesFolder, 'aug_img'))
