import tensorflow as tf
import tensorflow.contrib.slim.python.slim.preprocessing.vgg_preprocessing as vgg_preprocessing
# Reading data from TFRecord
def read_TFRecord(data_dir, batch_size, shuffle, in_classes,IMG_HEIGHT,IMG_WIDTH, is_training):
    num_classes = in_classes
    data_files = tf.gfile.Glob(data_dir)
    filename_queue = tf.train.string_input_producer(data_files,shuffle=True) 
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                       'img_width': tf.FixedLenFeature([], tf.int64),
                                       'img_height': tf.FixedLenFeature([], tf.int64),
                                       })  #get images and labels
    #decode images string--unit8
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    height = tf.cast(features['img_height'],tf.int32)
    width = tf.cast(features['img_width'],tf.int32)
    label = tf.cast(features['label'], tf.int32)
    channel = 3
    image = tf.reshape(image, [height,width,channel])
    image = vgg_preprocessing.preprocess_image(image,IMG_HEIGHT,IMG_WIDTH,is_training)
    print("image.shape",image)
    #image = tf.image.per_image_standardization(image)
    image = tf.cast(image, tf.float32)
    #generate batch
    min_after_dequeue = 2000
    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
                                             [image, label], 
                                             batch_size=batch_size,
                                             num_threads= 64,
                                             capacity=capacity, 
                                             min_after_dequeue=min_after_dequeue)
    else:
        image_batch, label_batch = tf.train.batch(
                                            [image, label], 
                                            batch_size=batch_size,
                                            num_threads = 64,
                                            capacity=capacity)
    ## ONE-HOT
    
    label_batch = tf.reshape(label_batch, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    label_batch = tf.sparse_to_dense(
                  tf.concat(values=[indices, label_batch], axis=1),
                  [batch_size, num_classes], 1.0, 0.0)
    print(image_batch)
    print(label_batch)
    
    #summar images for tensorboard
    #tf.summary.image('image_batch', image_batch)
    #return image_batch, label_batch

    #n_classes = 10
    #label_batch = tf.one_hot(label_batch, depth= n_classes)
    #label_batch = tf.cast(label_batch, dtype=tf.int32)
    #label_batch = tf.reshape(label_batch, [batch_size, n_classes])

    return image_batch, label_batch







