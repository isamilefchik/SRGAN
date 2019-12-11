import tensorflow as tf
import argparse

''' for calculating metrics '''

# convert filepath of image into tensor for metric calculations
def process_img(file_path):
    img = tf.io.read_file(file_path)

    if file_path.endswith(".png"):
        img = tf.io.decode_png(img, channels=3)
    elif file_path.endswith(".jpg"):
        img = tf.io.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.

    return img

# MSE
def calc_mse(true_img_path, output_img_path):
    true_img = process_img(true_img_path)
    output_img = process_img(output_img_path)

    mse = tf.keras.losses.MSE(true_img, output_img)
    return mse

# peak signal to noise
def calc_psnr(true_img_path, output_img_path):
    true_img = process_img(true_img_path)
    output_img = process_img(output_img_path)

    psnr = tf.image.psnr(true_img, output_img, max_val=1.0)
    return psnr

# structural similarity index metric
def calc_ssim(true_img_path, output_img_path):
    true_img = process_img(true_img_path)
    output_img = process_img(output_img_path)

    ssim = tf.image.ssim(true_img, output_img, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    return ssim

def main(args):
    # parse args for file paths
    print(calc_mse())
    print(calc_psnr())
    print(calc_ssim())

