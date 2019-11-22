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

    args_parser = argparse.ArgumentParser(description="Super Resolution GAN (SRGAN)")
    args_parser.add_argument("--load", \
            help='''Checkpoint directory to load model.''')
    args_parser.add_argument("--train", action='store_true', \
            help='''Trains the SRGAN model. Will create fresh model
            if one is not loaded.''')
    args_parser.add_argument("--infer", \
            help='''Run a loaded model on a given image. The argument passed
            in should be the path to the image to perform super resolution
            upon.''')
    args_parser.add_argument("--data", \
            help='''Path to folder of training data (folder containing
            .png images). Only full-scale images are needed. Images are
            resized when loaded in.''')
    args_parser.add_argument("--name", default="SRGAN", \
            help='''Name of the model (used for model saves).''')
    args_parser.add_argument("--epochs", type=int, default=1000, \
            help='''Number of epochs (for training only).''')

    args = args_parser.parse_args()

    gen_model = SRGAN_Generator(16)
    disc_model = SRGAN_Discriminator()

    checkpoint = tf.train.Checkpoint(gen_model=gen_model, disc_model=disc_model)
    checkpoint_manager = tf.train.CheckpointManager( \
            checkpoint, directory="./checkpoints", \
            max_to_keep=10, checkpoint_name=args.name)

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

    if args.train:
        if args.data is None:
            sys.exit("Must provide data folder for training.")
        train(gen_model, disc_model, args.data, args.epochs, checkpoint_manager)

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
