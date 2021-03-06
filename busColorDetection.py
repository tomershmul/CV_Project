import tensorflow as tf
import math
import os
import numpy as np
import re
from busColorParameters import *
from PIL import Image

def generate_dataset(folderPath, num_of_classes, size, Height, Width, color_dict_num):
    folder_file_names = [folderPath+i for i in os.listdir(folderPath)]
    folder_images_set = np.zeros((len(folder_file_names),Height,Width,3),dtype=np.float32)
    
    folder_image_labels = []
    idx=0
    for img in folder_file_names:
        image=Image.open(img)
        image = image.resize(size)
        #label = ''.join(filter(str.isalpha, os.path.basename(img).split('.')[0]))
        label = re.search('_([a-zA-Z]+)', os.path.basename(img).split('.')[0], re.IGNORECASE).group(1)
        folder_image_labels.append(color_dict_num[label])
        folder_images_set[idx,:,:,:] = np.array(image)
        idx+=1

    #Convert image labels to np.array
    folder_image_labels = np.array(folder_image_labels)
    folder_image_labels_onehot = np.zeros((len(folder_image_labels),num_of_classes))
    folder_image_labels_onehot[np.arange(folder_image_labels.shape[0]), folder_image_labels.astype(int)-1] = 1

    return folder_images_set, folder_image_labels_onehot

def generate_dataset_of_one(image, num_of_classes, size, Height, Width, color_dict_num):
    image_set = np.zeros((1,Height,Width,3),dtype=np.float32)
    image_label = 0 # We don't know the labal at this point
    idx=0 # only one image
    image = image.resize(size)
    image_set[0,:,:,:] = np.array(image)

    #Convert image labels to np.array # this part is redondant for predict...
    image_label = np.array(image_label)
    image_label_onehot = np.zeros((1,num_of_classes))
    image_label_onehot[0][0] = 1  # this part is redondant for predict...

    return image_set, image_label_onehot

    
def main():
    
    #Generate train & test dataset
    train_images_set, train_image_labels_onehot = generate_dataset(trainCropedBusesFolder, net_classes, size, Height, Width, color_dict_num)
    print ("Train Data set: "+str(train_images_set.shape[0])+" Croped-buses")
    test_images_set, test_image_labels_onehot = generate_dataset(testCropedBusesFolder, net_classes, size, Height, Width, color_dict_num)
    print ("Test Data set: "+str(test_images_set.shape[0])+" Croped-buses")
    
    #Define CNN 
    X = tf.placeholder(tf.float32, [None, Width, Height, 3], name="X")
    step = tf.placeholder(tf.int32, name="step")
    pkeep_fc = tf.placeholder(tf.float32, name="pkeep_fc")
    pkeep_conv = tf.placeholder(tf.float32, name="pkeep_conv")
    tf.add_to_collection('vars', pkeep_conv)
    tf.add_to_collection('vars', pkeep_fc)
    
    
    W1 = tf.Variable(tf.truncated_normal([3, 3, 3, out_L1] ,stddev=0.1), name="W1")
    B1 = tf.Variable(tf.truncated_normal([out_L1] ,stddev=0.1), name="B1")
    
    W2 = tf.Variable(tf.truncated_normal([3, 3, out_L1, out_L2] ,stddev=0.1), name="W2")
    B2 = tf.Variable(tf.truncated_normal([out_L2] ,stddev=0.1), name="B2")
    
    W3 = tf.Variable(tf.truncated_normal([3, 3, out_L2, out_L3] ,stddev=0.1), name="W3")
    B3 = tf.Variable(tf.truncated_normal([out_L3] ,stddev=0.1), name="B3")
    
    W4 = tf.Variable(tf.truncated_normal([7*7*out_L3, out_L4] ,stddev=0.1), name="W4")
    B4 = tf.Variable(tf.truncated_normal([out_L4] ,stddev=0.1), name="B4")
    
    W5 = tf.Variable(tf.truncated_normal([out_L4, net_classes], stddev=0.1), name="W5")
    B5 = tf.Variable(tf.truncated_normal([net_classes] ,stddev=0.1), name="B5")
    
    tf.add_to_collection('vars', W1)
    tf.add_to_collection('vars', W2)
    tf.add_to_collection('vars', W3)
    tf.add_to_collection('vars', W4)
    tf.add_to_collection('vars', W5)
    tf.add_to_collection('vars', B1)
    tf.add_to_collection('vars', B2)
    tf.add_to_collection('vars', B3)
    tf.add_to_collection('vars', B4)
    tf.add_to_collection('vars', B5)
    tf.add_to_collection('vars', X)
    
    # model
    with tf.name_scope("conv1"):
        stride = 1
        mp_stride = 2
        Ycnv1 = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
        Y1 = tf.nn.relu(Ycnv1 + B1)
        Y1d = tf.nn.dropout(Y1, pkeep_conv)
        Y1d = tf.nn.max_pool(Y1d, ksize=[1, mp_stride, mp_stride, 1], strides=[1, mp_stride, mp_stride, 1], padding='SAME')
    
    with tf.name_scope("conv2"):
        stride = 2
        #Ycnv2 = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
        Ycnv2 = tf.nn.conv2d(Y1d, W2, strides=[1, stride, stride, 1], padding='SAME')
        Y2 = tf.nn.relu(Ycnv2 + B2)
        Y2d = tf.nn.dropout(Y2, pkeep_conv)
        Y2d = tf.nn.max_pool(Y2d, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
    
    with tf.name_scope("conv3"):
        stride = 2
        #Ycnv3 = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
        Ycnv3 = tf.nn.conv2d(Y2d, W3, strides=[1, stride, stride, 1], padding='SAME')
        Y3 = tf.nn.relu(Ycnv3 + B3)
        Y3d = tf.nn.dropout(Y3, pkeep_conv)
        Y3d = tf.nn.max_pool(Y3d, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
    
        Y3_vec = tf.reshape(Y3d, shape=[-1,7*7*out_L3])
    
    with tf.name_scope("fc1"):
        Y4 = tf.nn.relu(tf.matmul(Y3_vec, W4) + B4)
        Y4d = tf.nn.dropout(Y4, pkeep_fc)
    
    with tf.name_scope("fc2"):
        Ylogits = tf.matmul(Y4, W5) + B5
        Y  = tf.nn.softmax(Ylogits, name="Y")
    
    tf.add_to_collection('vars', Y)
    
    
    # placeholder for correct labels
    Y_ = tf.placeholder(tf.float32, [None, net_classes], name="Y_")
    tf.add_to_collection('vars', Y_)
    
    with tf.name_scope("xent"):
        # loss function
        #cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
        cross_entropy = tf.reduce_mean(cross_entropy)*net_classes

        # Loss function with L2 Regularization with beta=0.01
        regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5)
        loss = tf.reduce_mean(cross_entropy + 0.01 * regularizers)
    
    with tf.name_scope("accuracy"):
        # % of correct answers found in batch
        is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    with tf.name_scope("train"):
        #optimizer = tf.train.GradientDescentOptimizer(0.003)
        #train_step = optimizer.minimize(cross_entropy)
        lr = 0.000001 + tf.train.exponential_decay(0.003, step, 100, 1/math.e)
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    saver = tf.train.Saver()
    
    
    for i in range(num_of_itr):
        train_data={X: train_images_set, Y_: train_image_labels_onehot, step: i, pkeep_conv: 0.7, pkeep_fc: 0.5}
        test_data={X: test_images_set, Y_: test_image_labels_onehot, pkeep_conv: 1.0, pkeep_fc: 1.0}
    
        # train
        sess.run(train_step, feed_dict=train_data)
        
        if i%10 ==0:
            a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
            print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))
        
        if i%100==0:
            a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
            print(str(i) + ": ********* epoch " + str(i) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        
    # success ?
    a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    print("Train: accuracy:" + str(a) + " loss: " + str(c))
    
    # success on test data ?
    a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    print("Test: accuracy:" + str(a) + " loss: " + str(c))
    
    classification = sess.run(Y, feed_dict=test_data)
    classification = np.argmax(classification, axis=1)
    print(classification)
    saver.save(sess, modelName, global_step=num_of_itr)
    
    
    #### tensor bord
    #with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs', sess.graph)
    #writer.add_graph(sess.graph)

    #print (sess.run(Y, feed_dict=test_data))
    
    writer.close()
    
if __name__ == "__main__":
    main()
