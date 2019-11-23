""" Preprocessing functions for SRGAN. """

import os
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_rand_crop(img, size=[96, 96, 3]):
    """ Takes in an image tensor, makes a random 96x96 crop of it,
    returns the crop and a 4x downsampled version of the crop.

    Args:
        img: Image tensor of shape [width, height, 3]
        size: List defining the shape of the crop (excluding
              the number of channels).

    Returns:
        A tuple of (rand_crop, small_rand_crop) where rand_crop is the
        image tensor cropped to the specified size and small_rand_crop
        is the image tensor cropped and 4x downsampled (bicubic).
    """

    # Get random crop
    rand_crop = tf.image.random_crop(img, size)

    # Get 4x bicubic downsampled version of crop
    small_rand_crop = tf.image.resize( \
            rand_crop, size=[24, 24], method=tf.image.ResizeMethod.BICUBIC)

    return rand_crop, small_rand_crop

def get_image_set(data_path, set_size):
    """ Gets training set of images from a directory for a single epoch.

    Args:
        data_path: path to directory of images
        set_size: number of images in set
        crop_size: size of crop

    Returns:
        tf.data.Dataset of image tensors
    """

    # Get file paths
    file_path_ds = tf.data.Dataset.list_files(os.path.join(data_path, "*.png"), shuffle=True)

    # Take only the size needed
    file_path_ds = file_path_ds.take(set_size)

    # Import the images and get crops
    img_ds = file_path_ds.map(_process_img_path, num_parallel_calls=AUTOTUNE)

    return img_ds

def _process_img_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.
    crop, small_crop = get_rand_crop(img)
    return small_crop, crop

def get_batches(img_ds, batch_size, cache=True):
    """ Basically stolen from Tensorflow documentation. Allows for
    efficient batching of images. """

    if cache:
        if isinstance(cache, str):
            img_ds = img_ds.cache(cache)
        else:
            img_ds = img_ds.cache()

    img_ds = img_ds.shuffle(buffer_size=1000)

    img_ds = img_ds.batch(batch_size)

    img_ds = img_ds.prefetch(buffer_size=AUTOTUNE)

    return img_ds
