#!/usr/local/bin/python3

""" Main routine of SRGAN program. """

import argparse
import sys
import os
from os import path
import tensorflow as tf

from model import SRGAN_Generator, SRGAN_Discriminator
from train import train_GAN
from infer import run_inference

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    """ Main function. """

    # Argument parsing
    args_parser = argparse.ArgumentParser(description="Super Resolution GAN (SRGAN)")
    args_parser.add_argument("--load", \
            help='''Checkpoint directory to load model.''')
    args_parser.add_argument("--train", \
            help='''The folder containing the data to train SRGAN on. If this
            flag is not used and a folder is not passed, then no training
            will occur. If a checkpoint folder is loaded using the --load flag,
            the model will first load the latest checkpoint then begin training.
            The data folder passed in should only contain the full-resolution images.
            The lower resolution images will be computed in the data processing step.
            Images should be in the .png format.''')
    args_parser.add_argument("--infer", \
            help='''Run a loaded model on a given image. The argument passed
            in should be the path to the image to be upsampled by the network.''')
    args_parser.add_argument("--name", default="SRGAN", \
            help='''Name of the model (used for model saves during training).''')
    args_parser.add_argument("--epochs", type=int, default=1000, \
            help='''Number of epochs (for training only).''')
    args = args_parser.parse_args()

    # Build models
    gen_model = SRGAN_Generator(16)
    disc_model = SRGAN_Discriminator()

    # Set up checkpoint and checkpoint_manager
    checkpoint = tf.train.Checkpoint(gen_model=gen_model, disc_model=disc_model)
    checkpoint_manager = tf.train.CheckpointManager( \
            checkpoint, directory="./checkpoints", \
            max_to_keep=10, checkpoint_name=args.name)

    # Load model if applicable
    if args.load is not None:
        if path.exists(args.load):
            if args.infer is not None and not args.train:
                checkpoint.restore( \
                        tf.train.latest_checkpoint(args.load)).expect_partial()
            else:
                checkpoint.restore( \
                        tf.train.latest_checkpoint(args.load))
        else:
            sys.exit("Checkpoint directory does not exist.")

    # Training
    if args.train:
        if args.train is None:
            sys.exit("Must provide data folder for training.")
        train(gen_model, disc_model, args.train, args.epochs, checkpoint_manager)

    # Inference
    if args.infer is not None:
        if path.exists(args.infer):
            infer(args.infer, gen_model)
        else:
            sys.exit("Inference image does not exist.")

def train(gen_model, disc_model, data, epochs, checkpoint_manager):
    """ Training routine. """

    for epoch in range(1, epochs + 1):
        print("============== EPOCH {} ==============".format(epoch))
        train_GAN(gen_model, disc_model, data)
        checkpoint_manager.save()

def infer(img_path, gen_model):
    """ Inference routine. """

    run_inference(img_path, gen_model, "output.png")

main()
