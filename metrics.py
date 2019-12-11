import tensorflow as tf
import argparse

''' for calculating metrics '''

# convert filepath of image into tensor for metric calculations
def process_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.
    return img

# MSE
def calc_mse(true_img, output_img):
    mse = tf.reduce_sum(tf.keras.losses.MSE(true_img, output_img))
    return mse

# peak signal to noise
def calc_psnr(true_img, output_img):
    psnr = tf.image.psnr(true_img, output_img, max_val=1.0)
    return psnr

# structural similarity index metric
def calc_ssim(true_img, output_img):
    ssim = tf.image.ssim(true_img, output_img, max_val=1.0, \
        filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    return ssim

def generate_bicubic(img):
    """ Runs inference on a single image. """

    # 4x upsample the small image
    bicubic_upsampling = tf.image.resize(img, [img.shape[0] * 4, img.shape[1] * 4], \
        method='bicubic')

    return bicubic_upsampling


def main():
    arg_parser = argparse.ArgumentParser( \
            description='''Runs metrics on a given image's SRGAN and
            bicubic upsampling''')
    arg_parser.add_argument("original", nargs=1, type=str, \
            help='''The path to the original small image that
            is fed into the SRGAN network.''')
    arg_parser.add_argument("gan", nargs=1, type=str, \
            help='''The path to the output of the SRGAN network.''')
    arg_parser.add_argument("truth", nargs=1, type=str, \
            help='''The path to the ground truth image.''')

    args = arg_parser.parse_args()

    original_image = args.original[0]
    gan_upsample = args.gan[0]
    truth = args.truth[0]

    # get images from filepaths
    small_image = process_img(original_image)
    bicubic_upsample = generate_bicubic(small_image)
    true_image = process_img(truth)
    gan_upsample = process_img(gan_upsample)

    print("Bicubic metrics:")
    print(calc_mse(true_image, bicubic_upsample))
    print(calc_psnr(true_image, bicubic_upsample))
    print(calc_ssim(true_image, bicubic_upsample))

    print("SRGAN metrics:")
    print(calc_mse(true_image, gan_upsample))
    print(calc_psnr(true_image, gan_upsample))
    print(calc_ssim(true_image, gan_upsample))

main()
