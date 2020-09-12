# -*-coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
#% matplotlib inline
import tensorflow.contrib.slim.python.slim.preprocessing.preprocessing_factory as preprocessing_factory
# import tensorflow.contrib.slim.python.slim.preprocessing.inception_preprocessing as inception_preprocessing
# import tensorflow.contrib.slim.python.slim.preprocessing.vgg_preprocessing as vgg_preprocessing
from tensorflow.python.framework import graph_util
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
vgg_size = 224
inception_size = 299
classes = os.listdir("D:/cnnmfacies/train/")


def predict(network_model, image_path):
    '''
    :param pb_path:pb file path
    :param image_path:tese image path
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with tf.Session() as sess:
            if network_model == 'inception_v4':
                image_size = inception_size
                pb_path = "inception_v4/inception_v4_frozen_model.pb"
            elif network_model == 'inception_resnet_v2':
                image_size = inception_size
                pb_path = "inception_resnet_v2/inception_resnet_v2_frozen_model.pb"
            elif network_model == 'vgg_16':
                image_size = vgg_size
                pb_path = "vgg_16/vgg_16_frozen_model.pb"
            elif network_model == 'resnet_v1_152':
                image_size = vgg_size
                pb_path = "resnet_v1_152/resnet_v1_152_frozen_model.pb"
            else:
                print("invalid network model : " + network_model)

            with open(pb_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            sess.run(tf.global_variables_initializer())

            if network_model == 'inception_v4':
                output_tensor_name = sess.graph.get_tensor_by_name("InceptionV4/Logits/Predictions:0")
                image_preprocessing_fn = preprocessing_factory.get_preprocessing('inception_v4', is_training=False)
            elif network_model == 'inception_resnet_v2':
                output_tensor_name = sess.graph.get_tensor_by_name("InceptionResnetV2/Logits/Predictions:0")
                image_preprocessing_fn = preprocessing_factory.get_preprocessing('inception_resnet_v2', is_training=False)
            elif network_model == 'vgg_16':
                output_tensor_name = sess.graph.get_tensor_by_name("vgg_16/fc8/squeezed:0")
                image_preprocessing_fn = preprocessing_factory.get_preprocessing('vgg_16', is_training=False)
            elif network_model == 'resnet_v1_152':
                output_tensor_name = sess.graph.get_tensor_by_name("resnet_v1_152/logits/predictions:0")
                image_preprocessing_fn = preprocessing_factory.get_preprocessing('resnet_v1_152', is_training=False)
            else:
                print("invalid network model : " + network_model)

            # input:0 as input_images,keep_prob:0 as  dropout=1.0 for test,is_training:0 as False for test
            input_image_tensor = sess.graph.get_tensor_by_name("input_images:0")
            input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            image_data = tf.image.decode_jpeg(image_data, channels=3)
            img = np.asarray(image_data.eval(), dtype='uint8')
            plt.imshow(img)
            # plt.axis('off')
            plt.show()

            image_data = image_preprocessing_fn(image_data, image_size, image_size)
            image_data = tf.image.per_image_standardization(image_data)
            image_data = tf.expand_dims(image_data, 0)
            image_data = np.asarray(image_data.eval(), dtype='float32')
            out = sess.run(output_tensor_name, feed_dict={input_image_tensor: image_data,
                                                          input_keep_prob_tensor: 1.0,
                                                          input_is_training_tensor: False})
            print("out:{}".format(out))
            # predictions = tf.nn.softmax(out, name='pre')
            predictions = np.squeeze(out)
            top_k = predictions.argsort()[::-3]
            # print top_3 possibilities
            for m in range(3):
                class_name = classes[top_k[m]]
                score = predictions[top_k[m]]
                print('%s (Possibility = %.2f%%)' % (class_name, score * 100))


if __name__ == '__main__':
    # expect networks "vgg_16", "resnet_v1_152", "inception_v4", "inception_resnet_v2"
    # image_path = 'D:/cnnmfacies/test/algae/algae-10040-21(3.44_LFY.png'
    image_path = 'D:/cnnmfacies/test/bryozoan/bryozoan-fig2-9_Ernst-2009.png'
    network_model = 'inception_v4'
    predict(network_model, image_path=image_path)