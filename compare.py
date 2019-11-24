from matplotlib import pyplot as plt
from skimage.transform import resize
import random
import argparse
from PIL import Image
import numpy as np

def make_visual_comparison(original_image, output_image):

    original_w, original_h = original_image.size

    bicubic_resize = original_image.resize( \
            (original_w * 4, original_h * 4), resample=Image.BICUBIC)

    original_image = np.array(original_image)
    bicubic_resize = np.array(bicubic_resize)
    output_image = np.array(output_image)

    num_examples = 2
    crop_size = 40

    fig = plt.figure(figsize=(10, 8))

    for i in range(1, num_examples*3 + 1, 3):

        # Get crop ranges
        left = random.randrange(0, original_w - crop_size)
        top = random.randrange(0, original_h - crop_size)
        right = left + crop_size
        bottom = top + crop_size

        fig.add_subplot(num_examples, 3, i)
        original_crop = original_image[top:bottom, left:right]
        plt.axis('off')
        plt.imshow(original_crop)

        fig.add_subplot(num_examples, 3, i+1)
        bicubic_crop = bicubic_resize[top*4:bottom*4, left*4:right*4]
        plt.axis('off')
        plt.imshow(bicubic_crop)

        fig.add_subplot(num_examples, 3, i+2)
        output_crop = output_image[top*4:bottom*4, left*4:right*4]
        plt.axis('off')
        plt.imshow(output_crop)

    plt.show()

def main():
    arg_parser = argparse.ArgumentParser( \
            description="Visualization tool for SRGAN results.")
    arg_parser.add_argument("original", nargs=1, type=str, \
            help='''The path to the original image.''')
    arg_parser.add_argument("output", nargs=1, type=str, \
            help='''The path to the output image of SRGAN given the
            original image.''')
    args = arg_parser.parse_args()

    original_image = Image.open(args.original[0])
    output_image = Image.open(args.output[0])

    make_visual_comparison(original_image, output_image)

main()
