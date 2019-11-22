""" Inference functions for SRGAN. """

import tensorflow as tf

def run_inference(img_path, gen_model, output_path):
    """ Runs inference on a single image. """

    # Read file
    img_file = tf.io.read_file(img_path)
    img = tf.io.decode_png(img_file, channels=3)
    img = tf.cast(img, tf.float32) / 255.

    # Make a batch of one image
    img = tf.expand_dims(img, 0)

    # Generate super resolution version
    output = tf.squeeze(gen_model(img))

    # Export png
    output = tf.clip_by_value(output * 255., 0., 255.)
    output = tf.cast(output, tf.uint8)
    output = tf.image.encode_png(output)
    tf.io.write_file(output_path, output)
