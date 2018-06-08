import tensorflow as tf
import math
import os
import numpy as np
from PIL import Image

def generate_dataset(folderPath, num_of_classes, size, Height, Width, color_dict_num):
    folder_file_names = [folderPath+i for i in os.listdir(folderPath)]
    folder_images_set = np.zeros((len(folder_file_names),Height,Width,3),dtype=np.float32)
    
    folder_image_labels = []
    idx=0
    for img in folder_file_names:
        image=Image.open(img)
        image = image.resize(size)
        label = ''.join(filter(str.isalpha, os.path.basename(img).split('.')[0]))
        folder_image_labels.append(color_dict_num[label])
        folder_images_set[idx,:,:,:] = np.array(image)
        idx+=1

    #Convert image labels to np.array
    folder_image_labels = np.array(folder_image_labels)
    folder_image_labels_onehot = np.zeros((len(folder_image_labels),num_of_classes))
    folder_image_labels_onehot[np.arange(folder_image_labels.shape[0]), folder_image_labels.astype(int)-1] = 1

    return folder_images_set, folder_image_labels_onehot

def main():
    color_dict = {'1': 'Green', '2': 'Yellow', '3': 'White', '4': 'Grey', '5': 'Blue', '6': 'Red'}
    color_dict_num = {'Green': 1, 'Yellow': 2, 'White': 3, 'Grey': 4, 'Blue': 5, 'Red': 6}
    
    trainFolderPath = os.path.abspath(os.path.join(os.path.curdir,'train'))
    trainFolderPath = os.path.abspath(os.path.join(trainFolderPath, 'Croped_buses'))
    modelPath = os.path.abspath(os.path.join(trainFolderPath, 'Model'))
    modelName = os.path.abspath(os.path.join(modelPath, 'busColorDetector'))
    
    trainFolder = os.path.abspath(os.path.join(trainFolderPath, 'train'))
    testFolder = os.path.abspath(os.path.join(trainFolderPath, 'test'))
    
    trainFolder = trainFolder +"/"#+ '\\'
    testFolder = testFolder +"/"#+ '\\'
    
    #CNN Parameters
    out_L1=6
    out_L2=6
    out_L3=6
    out_L4=50 #Fully Connected layer
    
    net_classes=6
    num_of_itr=300
    
    
    #Resize Parameters
    Width = 224
    Height = 224
    size = Width, Height
    
    #Generate train & test dataset
    train_images_set, train_image_labels_onehot = generate_dataset(trainFolder, net_classes, size, Height, Width, color_dict_num)
    test_images_set, test_image_labels_onehot = generate_dataset(testFolder, net_classes, size, Height, Width, color_dict_num)
    
    #Define CNN 
    X = tf.placeholder(tf.float32, [None, Width, Height, 3], name="X")
    step = tf.placeholder(tf.int32, name="step")
    pkeep = tf.placeholder(tf.float32, name="pkeep")
    tf.add_to_collection('vars', pkeep)
    
    
    W1 = tf.Variable(tf.truncated_normal([1, 1, 3, out_L1] ,stddev=0.1), name="W1")
    B1 = tf.Variable(tf.ones([out_L1])/net_classes, name="B1")
    
    W2 = tf.Variable(tf.truncated_normal([3, 3, out_L1, out_L2] ,stddev=0.1), name="W2")
    B2 = tf.Variable(tf.ones([out_L2])/net_classes, name="B2")
    
    W3 = tf.Variable(tf.truncated_normal([3, 3, out_L2, out_L3] ,stddev=0.1), name="W3")
    B3 = tf.Variable(tf.ones([out_L3])/net_classes, name="B3")
    
    W4 = tf.Variable(tf.truncated_normal([7*7*out_L3, out_L4] ,stddev=0.1), name="W4")
    B4 = tf.Variable(tf.ones([out_L4])/net_classes, name="B4")
    
    W5 = tf.Variable(tf.truncated_normal([out_L4, net_classes], stddev=0.1), name="W5")
    B5 = tf.Variable(tf.ones([net_classes])/net_classes, name="B5")
    
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
    
    stride = 2
    Ycnv1 = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
    Y1 = tf.nn.relu(Ycnv1 + B1)
    Y1d = tf.nn.dropout(Y1, pkeep)
    Y1d = tf.nn.max_pool(Y1d, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
    
    stride = 2
    #Ycnv2 = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
    Ycnv2 = tf.nn.conv2d(Y1d, W2, strides=[1, stride, stride, 1], padding='SAME')
    Y2 = tf.nn.relu(Ycnv2 + B2)
    Y2d = tf.nn.dropout(Y2, pkeep)
    Y2d = tf.nn.max_pool(Y2d, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
    
    stride = 2
    #Ycnv3 = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
    Ycnv3 = tf.nn.conv2d(Y2d, W3, strides=[1, stride, stride, 1], padding='SAME')
    Y3 = tf.nn.relu(Ycnv3 + B3)
    Y3d = tf.nn.dropout(Y3, pkeep)
    
    Y3_vec = tf.reshape(Y3, shape=[-1,7*7*out_L3])
    
    Y4 = tf.nn.relu(tf.matmul(Y3_vec, W4) + B4)
    Y4d = tf.nn.dropout(Y4, pkeep)
    
    Ylogits = tf.matmul(Y4, W5) + B5
    Y  = tf.nn.softmax(Ylogits, name="Y")
    
    tf.add_to_collection('vars', Y)
    
    
    # placeholder for correct labels
    Y_ = tf.placeholder(tf.float32, [None, net_classes], name="Y_")
    tf.add_to_collection('vars', Y_)
    
    # loss function
    #cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*net_classes
    
    # % of correct answers found in batch
    is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    #optimizer = tf.train.GradientDescentOptimizer(0.003)
    #train_step = optimizer.minimize(cross_entropy)
    
    lr = 0.0001 + tf.train.exponential_decay(0.003, step, 100, 1/math.e)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    saver = tf.train.Saver()
    
    
    for i in range(num_of_itr):
        train_data={X: train_images_set, Y_: train_image_labels_onehot, step: i, pkeep: 0.5}
        test_data={X: test_images_set, Y_: test_image_labels_onehot, pkeep: 1.0}
    
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
    
    saver.save(sess, modelName, global_step=num_of_itr)
    
if __name__ == "__main__":
    main()


#
##sess2 = tf.Session()
#new_graph = tf.Graph()
#metaFile = modelName + '-1000.meta'
#
#with tf.Session(graph=new_graph) as sess2:
#    sess2.run(init)
#    new_saver = tf.train.import_meta_graph(metaFile)
#    new_saver.restore(sess2, tf.train.latest_checkpoint(modelPath))
#    all_vars = tf.get_collection('vars')
#    for v in all_vars:
#        v_ = sess2.run(v)
#        print(v_)
#    
