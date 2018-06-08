import tensorflow as tf
import math
import os
import numpy as np
from PIL import Image
from busColorDetection import generate_dataset


color_dict = {'1': 'Green', '2': 'Yellow', '3': 'White', '4': 'Grey', '5': 'Blue', '6': 'Red'}
color_dict_num = {'Green': 1, 'Yellow': 2, 'White': 3, 'Grey': 4, 'Blue': 5, 'Red': 6}

trainFolderPath = os.path.abspath(os.path.join(os.path.curdir,'train'))
trainFolderPath = os.path.abspath(os.path.join(trainFolderPath, 'Croped_buses'))

trainFolder = os.path.abspath(os.path.join(trainFolderPath, 'train'))
testFolder = os.path.abspath(os.path.join(trainFolderPath, 'test'))

modelPath = os.path.abspath(os.path.join(trainFolderPath, 'Model'))
modelPath = modelPath + '/' #'\\'
modelName = os.path.abspath(os.path.join(modelPath, 'busColorDetector'))
metaFile = modelName + '-300.meta'
#ckptfile = modelPath + ''

net_classes=6
Width = 224
Height = 224
size = Width, Height

testFolder = testFolder + '/' #'\\'

#new_graph = tf.Graph()
new_graph = tf.get_default_graph()
#X = tf.placeholder(tf.float32, [None, Width, Height, 3])
#step = tf.placeholder(tf.int32)
#pkeep = tf.placeholder(tf.float32)
#Y = tf.placeholder(tf.float32, [None, 6])
#Y_ = tf.placeholder(tf.float32, [None, 6])


#with tf.Session(graph=new_graph) as sess:
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    new_saver = tf.train.import_meta_graph(metaFile)
    new_saver.restore(sess, tf.train.latest_checkpoint(modelPath))
#    new_saver.restore(sess, modelPath)
#    new_saver=tf.train.Saver()
#    new_saver.restore(sess, metaFile)
#    all_vars = tf.get_collection('vars')
    all_vars = tf.get_collection('vars')
    for v in all_vars:
        print(v)
    
    X=new_graph.get_tensor_by_name("X:0")
    Y=new_graph.get_tensor_by_name("Y:0")
    Y_=new_graph.get_tensor_by_name("Y_:0")
    pkeep=new_graph.get_tensor_by_name("pkeep:0") 
    #tf.get_collection('vars')
    test_images_set, test_image_labels_onehot = generate_dataset(testFolder, net_classes, size, Height, Width, color_dict_num)
    test_data={X: test_images_set, Y_: test_image_labels_onehot, pkeep: 1.0}
#    for v in all_vars:
#        v_ = sess.run(v)
#        print(v_)
    #classification = sess.run(Y, feed_dict={X: test_images_set})
    classification = sess.run(Y, feed_dict=test_data)
    max_index = np.argmax(classification, axis=1)
    print (max_index)
