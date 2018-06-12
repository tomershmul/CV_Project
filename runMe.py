import numpy as np
import ast
import sys
import os
from jsonToAnnotations import json_to_ann
from cropBuses import crop_buses

def run(myAnnFileName, buses):
    #tiny-yolo-1c-thresh0p7-bias0.cfg
    myConfig = "my_yolo-1c.cfg"
    myConfigDir = os.path.abspath(os.path.join(os.path.curdir,'cfg'))
    myConfig = os.path.abspath(os.path.join(myConfigDir,myConfig))
    #flowCmd = "flow --model " +myConfig+ " --load -1 --imgdir " +buses+ " --labels labels_1c.txt  --threshold 0.6 --json"
    flowCmd = "flow --pbLoad built_graph/my_yolo-1c.pb --metaLoad built_graph/my_yolo-1c.meta  --json --threshold 0.6 "+" --imgdir " +buses
    flowCmd = "flow --pbLoad built_graph/my_yolo-1c.9Jun.pb --metaLoad built_graph/my_yolo-1c.9Jun.meta  --json --threshold 0.6 "+" --imgdir " +buses
    os.system(flowCmd)
    yoloOutDir = os.path.abspath(os.path.join(buses, "out"))
    #yoloOutDir = buses+"/out"
    #jsonCmd = "python3 jsonToAnnotations.py -myTestFolderPath " + buses+"/out -myTestAnnFileName " +myAnnFileName
    #os.system(jsonCmd)
    json_to_ann(yoloOutDir, buses, myAnnFileName)
    #cropBusesCmd = ""
    #annotationsFilePath = os.path.join(os.getcwd(), myAnnFileName)
    #crop_buses(annotationsFilePath, buses)
    
