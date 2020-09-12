# -*- coding: UTF-8 -*-
# Created by Xiaokang Liu in China University of Geosciences, Wuhan, China, Sep. 2020
import os.path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v4 as inception_v4
import tensorflow.contrib.slim.python.slim.preprocessing.inception_preprocessing as inception_preprocessing

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Structure of prediction directories expected
# |── test/
# |   |── algae
# |   |   |── algae_img1.png
# |   |   |── algae_img2.png
# |   |   |── ...
# |   |── bivalve
# |   |   |── bivalve_img1.png
# |   |   |── bivalve_img2.png
# |   |   |── ...
# |   |── ...
test = "D:/cnnmfacies/test/"
classes = os.listdir(test)
HEIGHT = 299
WIDTH = 299
is_training = False
inputs = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='input_images')
with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    logits, _ = inception_v4.inception_v4(inputs, num_classes=22, is_training=is_training, dropout_keep_prob=1.0)
final_tensor = tf.nn.softmax(logits)


def predict():
    with tf.Session() as sess:
        tf.train.Saver().restore(sess,'D:/netslogs/inception_v4/inception_v4/save_model/model.ckpt-40')
        inputs = sess.graph.get_tensor_by_name('input_images:0')
        for root, dirs, files in os.walk(test):
            strings = dirs
            for file in files:
                image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
                image_data = tf.image.decode_jpeg(image_data, channels=3)
                img = np.asarray(image_data.eval(), dtype='uint8')
                plt.imshow(img)
                # plt.axis('off')
                plt.show()
                image_data = inception_preprocessing.preprocess_for_eval(image_data, HEIGHT, WIDTH, is_training)
                # image_data = inception_preprocessing.preprocess_for_train(image_data,HEIGHT,WIDTH,None,random_crop=False)
                # image_data = tf.image.per_image_standardization(image_data)
                image_data = tf.expand_dims(image_data, 0)
                image_data = np.asarray(image_data.eval(), dtype='float32')
                predictions = sess.run(final_tensor, feed_dict={inputs: image_data})
                predictions = np.squeeze(predictions)
                image_path = os.path.join(root, file)
                print(image_path)
                top_k = predictions.argsort()[::-3]
                print("top_k", top_k)
                for m in range(3):  # print top_3 possibilities
                    class_name = classes[top_k[m]]
                    score = predictions[top_k[m]]
                    print('%s (Possibility = %.2f%%)' % (class_name, score * 100))
                print()
predict()
