import tensorflow as tf
import numpy as np
import Image
import os
def off_warning():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def getBatch(i, size, trainFeatures, trainLabels):
    startIndex = (i * size)
    endIndex = startIndex + size
    batch_X = trainFeatures[startIndex : endIndex]
    batch_Y = trainLabels[startIndex : endIndex]
    return batch_X, batch_Y


def CNN(x, keep_prob):

    #--->> Filter
    #shape is related to one anathor filter
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 5], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 5, 7], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 7, 6], mean=0, stddev=0.08))
    conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 5,1], mean=0, stddev=0.08))

    # -> conv2d() layer
    # -> activation function(relu)
    # -> max_polling (to reduce size of input)
    # -> bath norm

    #1st layer
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 2nd
    # conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
    # conv2 = tf.nn.relu(conv2)
    # conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv2_bn = tf.layers.batch_normalization(conv2_pool)
    #
    # # 3rd
    # conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1, 1, 1, 1], padding='SAME')
    # conv3 = tf.nn.relu(conv3)
    # conv3_pool = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv3_bn = tf.layers.batch_normalization(conv3_pool)

    # 4th
    conv4 = tf.nn.conv2d(conv1_bn, conv4_filter, strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv4_bn = tf.layers.batch_normalization(conv4_pool)

    # covert input for fully connected layer
    flat = tf.contrib.layers.flatten(conv4_bn)

    #fully connected layer
    #-> Fully_connected layer
    #-> dropout (for regulerization)
    #->batch norm
    # 1st
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    # # 2nd
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)

    # 3rd
    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep_prob)
    full3 = tf.layers.batch_normalization(full3)

    # 4th
    full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
    full4 = tf.nn.dropout(full4, keep_prob)
    full4 = tf.layers.batch_normalization(full4)

    # final layer (with+out actication function)
    out = tf.contrib.layers.fully_connected(inputs=full4, num_outputs=25, activation_fn=None)
    return out

def plot_a_pic(test_X):
    import matplotlib.pyplot as plt
    pixels = test_X.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def get_image(image_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    image = Image.open(image_path, 'r')
    width, height = image.size
    pixel_values = list(image.getdata())
    if image.mode == 'RGB':
        channels = 3
    elif image.mode == 'L':
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = np.array(pixel_values).reshape((1,width, height, channels))
   # print(pixel_values.shape)
    return pixel_values


def one_hot(list,size):
    x=np.zeros((len(list),size))
    j=0
    for i in list:
            x[j,i]=1
            j+=1
    return x

def save_model(sess,name):
    saver = tf.train.Saver()
    saver.save(sess,"model/{}".format(name))
    print("Model saved")

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname( 'final_model/'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot -> {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

    else:
        print("Initializing fresh parameters for the Chatbot")










def cnn_fun(input_layer):

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.relu,
        name="conv1")
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="conv2")

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=5, name="pool2")


    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="conv3")


    # Dense Layer
    flat = tf.contrib.layers.flatten(conv3)

    dense = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu, name="dense")

    dropout = tf.layers.dropout(inputs=dense, rate=0.2, name="dropout")

    # Logits Layer
    num_of_classes = 25
    logits = tf.layers.dense(inputs=dropout, units=num_of_classes, name="logits")

    return logits