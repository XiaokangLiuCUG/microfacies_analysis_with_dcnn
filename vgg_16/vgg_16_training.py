# -*- coding: UTF-8 -*-
# Created by Xiaokang Liu in China University of Geosciences, Wuhan, China, in Sep. 2020
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import input_data
from time import time
import tensorflow.contrib.slim.python.slim.nets.vgg as vgg
# from tensorflow.python.framework import graph_util
# from tensorflow.python.platform import gfile
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

START_LEARNING_RATE = 0.0001
DECAY_STEPS = 400
DECAY_RATE = 0.96
BATCH_SIZE = 50
N_CLASSES = 22
HEIGHT = 224
WIDTH = 224
epochs = 50
each_epoch_step = int(24661 / BATCH_SIZE)
MAX_STEPS = each_epoch_step * epochs
# test parameters
n_test = 1530
num_batch = int(n_test / BATCH_SIZE + 1)
num_sample = num_batch * BATCH_SIZE

CKPT_FILE = "./vgg_16.ckpt"
CHECKPOINT_EXCLUDE_SCOPES = 'vgg_16/fc6,vgg_16/fc7,vgg_16/fc8'
TRAINABLE_SCOPES = 'vgg_16/conv4,vgg_16/conv5,vgg_16/fc6,vgg_16/fc7,vgg_16/fc8'
TRAIN_FILE = "D:/netslogs/VGG16/save_model/"
VALIDATION_FILE = "D:/netslogs/VGG16/validation_log/"
train_data_dir = 'D:/mfaciesdatatfrecord/train/traindata.tfrecords*'
validation_data_dir = 'D:/mfaciesdatatfrecord/validation/validationdata.tfrecords*'
test_data_dir = 'D:/mfaciesdatatfrecord/test/testdata.tfrecords*'

'''
  expected_names = ['vgg_16/conv1/conv1_1',
                    'vgg_16/conv1/conv1_2',
                    'vgg_16/pool1',
                    'vgg_16/conv2/conv2_1',
                    'vgg_16/conv2/conv2_2',
                    'vgg_16/pool2',
                    'vgg_16/conv3/conv3_1',
                    'vgg_16/conv3/conv3_2',
                    'vgg_16/conv3/conv3_3',
                    'vgg_16/pool3',
                    'vgg_16/conv4/conv4_1',
                    'vgg_16/conv4/conv4_2',
                    'vgg_16/conv4/conv4_3',
                    'vgg_16/pool4',
                    'vgg_16/conv5/conv5_1',
                    'vgg_16/conv5/conv5_2',
                    'vgg_16/conv5/conv5_3',
                    'vgg_16/pool5',
                    'vgg_16/fc6',
                    'vgg_16/fc7',
                    'vgg_16/fc8'
                   ]
'''


# Load pretrain data
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(",")]
    # print("exclusions",exclusions)
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


# Get all trainable_variables
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(",")]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main():
    startTime = time()
    # read train data
    training_images, training_labels = input_data.read_TFRecord(data_dir=train_data_dir,
                                                                batch_size=BATCH_SIZE,
                                                                shuffle=True,
                                                                in_classes=N_CLASSES,
                                                                IMG_HEIGHT=HEIGHT,
                                                                IMG_WIDTH=WIDTH,
                                                                is_training=True)
    # read validation data
    validation_images, validation_labels = input_data.read_TFRecord(data_dir=validation_data_dir,
                                                                    batch_size=BATCH_SIZE,
                                                                    shuffle=True,
                                                                    in_classes=N_CLASSES,
                                                                    IMG_HEIGHT=HEIGHT,
                                                                    IMG_WIDTH=WIDTH,
                                                                    is_training=False)

    # read test data
    test_images, test_labels = input_data.read_TFRecord(data_dir=test_data_dir,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False,
                                                        in_classes=N_CLASSES,
                                                        IMG_HEIGHT=HEIGHT,
                                                        IMG_WIDTH=WIDTH,
                                                        is_training=False)
    sess = tf.Session()

    images = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='input_images')
    labels = tf.placeholder(tf.int64, shape=[None, N_CLASSES], name='labels')
    keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
    global_step = tf.Variable(0, trainable=False)
    is_training = tf.placeholder(tf.bool, name='is_training')
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(images, num_classes=N_CLASSES, is_training=is_training, dropout_keep_prob=keep_prob)

    # Get_trainable_variables
    trainable_variables = get_trainable_variables()
    # print("get_trainable_variables",trainable_variables)
    # print("all_trainable_variables",tf.trainable_variables())
    tf.losses.softmax_cross_entropy(labels, logits, weights=1.0)
    loss = tf.losses.get_total_loss()
    tf.summary.scalar('loss', loss)
    learning_rate = tf.train.exponential_decay(START_LEARNING_RATE, global_step, DECAY_STEPS, DECAY_RATE,
                                               staircase=True)
    # Only training few or all layers
    # train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss,global_step=global_step,var_list=trainable_variables)
    train_step = tf.train.AdamOptimizer(START_LEARNING_RATE).minimize(loss, global_step=global_step,
                                                                      var_list=trainable_variables)
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct_prediction_top3 = tf.nn.in_top_k(logits, tf.argmax(labels, 1), 3)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        evaluation_step_top3 = tf.reduce_mean(tf.cast(correct_prediction_top3, tf.float32))
        correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
        correct_num_top3 = tf.reduce_sum(tf.cast(correct_prediction_top3, tf.int32))
        tf.summary.scalar('accuracy', evaluation_step)
        tf.summary.scalar('accuracy_top3', evaluation_step_top3)

    # Load pretrain weight
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), ignore_missing_vars=True)

    # Check saved model
    startepoch = tf.Variable(0, name='startepoch', trainable=False)
    epoch_start_time = time()
    saver = tf.train.Saver(tf.global_variables())
    # Save model
    # saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print("Load pretrain data")
        load_fn(sess)

        if not os.path.exists(TRAIN_FILE):
            os.makedirs(TRAIN_FILE)
        saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.latest_checkpoint(TRAIN_FILE)
        if ckpt != None:
            saver.restore(sess, ckpt)
        else:
            print("Training from scratch.")
        start = sess.run(startepoch)
        print("Training starts from {} epoch.".format(start + 1))

        tra_summary_writer = tf.summary.FileWriter(TRAIN_FILE, sess.graph)
        val_summary_writer = tf.summary.FileWriter(VALIDATION_FILE, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for epoch in range(start, epochs):
                for step in range(0, each_epoch_step):
                    if coord.should_stop():
                        break
                    global_steps = sess.run(global_step)
                    # training
                    tra_images, tra_labels = sess.run([training_images, training_labels])
                    _, train_loss, train_accuracy, train_accuracy_top3 = sess.run(
                        [train_step, loss, evaluation_step, evaluation_step_top3],
                        feed_dict={images: tra_images, labels: tra_labels, keep_prob: 0.8,is_training:True})
                    if (global_steps + 1) % 500 == 0 or (global_steps + 1) == MAX_STEPS:
                        tra_images, tra_labels = sess.run([training_images, training_labels])
                        summary_str, train_loss, train_accuracy, train_accuracy_top3 = sess.run([summary_op, loss, evaluation_step, evaluation_step_top3],
                                                                                                feed_dict={images: tra_images, labels: tra_labels, keep_prob: 0.8,is_training:False})
                        tra_summary_writer.add_summary(summary_str, global_steps + 1)
                        print("After %d training steps，train_loss = %.4f,train_accuracy = %.2f%%,train_accuracy_top3 = %.2f%%" %
                            (global_steps + 1, train_loss, train_accuracy * 100, train_accuracy_top3 * 100))

                    # validation
                    if (global_steps + 1) % 500 == 0 or (global_steps + 1) == MAX_STEPS:
                        val_images, val_labels = sess.run([validation_images, validation_labels])
                        summary_str, validation_loss, validation_accuracy, validation_accuracy_top3 = sess.run([summary_op, loss, evaluation_step, evaluation_step_top3],
                                                                                                               feed_dict={images: val_images, labels: val_labels, keep_prob: 0.8,is_training:False})
                        val_summary_writer.add_summary(summary_str, global_steps + 1)
                        print("After %d training steps，validation_loss = %.4f,validation_accuracy = %.2f%%,validation_accuracy_top3 = %.2f%%" %
                            (global_steps + 1, validation_loss, validation_accuracy * 100,
                             validation_accuracy_top3 * 100)) \
                            # test
                if (epoch + 1) % 2 == 0 or (epoch + 1) == epochs:
                    total_correct = 0
                    total_correct_top3 = 0
                    for each_bach in range(num_batch):
                        tes_images, tes_labels = sess.run([test_images, test_labels])
                        batch_correct, batch_correct_top_3 = sess.run([correct_num, correct_num_top3],
                                                                      feed_dict={images: tes_images, labels: tes_labels, keep_prob: 1.0,is_training:False})
                        # print("batch_correct:",batch_correct)
                        total_correct += np.sum(batch_correct)
                        total_correct_top3 += np.sum(batch_correct_top_3)
                    print('Total testing samples: %d' % num_sample)
                    print('Top_1 test average accuracy: %.2f%%' % (100 * total_correct / num_sample))
                    print('Top_3 test average accuracy: %.2f%%' % (100 * total_correct_top3 / num_sample))
                # save model
                if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                    checkpoint_path = os.path.join(TRAIN_FILE, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=epoch + 1)

                epoch_end_time = time()
                print("Current Epoch Takes:%.2f" % (epoch_end_time - epoch_start_time))
                epoch_start_time = epoch_end_time
                # check save point
                sess.run(startepoch.assign(epoch + 1))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)
        # save as pb form
        '''
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["vgg_16/fc8/squeezed:0"])
        with tf.gfile.FastGFile(os.path.join(TRAIN_FILE,'inception_v4_model.pb'), mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        '''
        duration = time() - startTime
        print("Total Train Takes:", duration)


if __name__ == '__main__':
    main()