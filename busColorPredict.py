import tensorflow as tf
import math
import os
import numpy as np
from PIL import Image
from busColorParameters import *
from busColorDetection import generate_dataset


def predict(test_images_set, test_image_labels_onehot):
    new_graph = tf.get_default_graph()
    
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(metaFile)
        new_saver.restore(sess, tf.train.latest_checkpoint(modelPath))
        #all_vars = tf.get_collection('vars')
        #for v in all_vars:
        #    print(v)
        
        X=new_graph.get_tensor_by_name("X:0")
        Y=new_graph.get_tensor_by_name("Y:0")
        Y_=new_graph.get_tensor_by_name("Y_:0")
        pkeep=new_graph.get_tensor_by_name("pkeep:0") 
        test_data={X: test_images_set, Y_: test_image_labels_onehot, pkeep: 1.0}
        #classification = sess.run(Y, feed_dict={X: test_images_set})
        classification = sess.run(Y, feed_dict=test_data)
        max_index = np.argmax(classification, axis=1)
        print (max_index)
        return(max_index)

if __name__ == "__main__":
    test_images_set, test_image_labels_onehot = generate_dataset(testFolder, net_classes, size, Height, Width, color_dict_num)
    max_index = predict(test_images_set, test_image_labels_onehot)
    print (max_index)
    print (test_image_labels_onehot)
    

