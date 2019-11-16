""" Defines the model architecture of the generator and discriminator
networks of SRGAN as well as their loss functions. """

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.applications.vgg19 as vgg19

# ===========================================================================
# GENERATOR
# ===========================================================================

class SRGAN_Generator(tf.keras.Model):
    """ Defines the generator network of SRGAN. """

    def __init__(self, B):
        super(SRGAN_Generator, self).__init__(name='')

        # Loss:
        # ===============

        self.content_loss_model = self._init_content_loss_model()

        # Layers:
        # ===============

        # Pre-residual encoding
        self.conv1 = layers.Conv2D( \
                filters=64, kernel_size=9, strides=1, padding='same')
        self.prelu1 = layers.PReLU(shared_axes=[1, 2])

        # Res-blocks
        self.resblocks = []
        for _ in range(B):
            self.resblocks.append(GenResBlock())

        # Final residual connection block
        self.conv2 = layers.Conv2D( \
                filters=64, kernel_size=3, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        # Pixel shuffle 1
        self.conv3 = layers.Conv2D( \
                filters=256, kernel_size=3, strides=1, padding='same')
        self.prelu3 = layers.PReLU(shared_axes=[1, 2])

        # Pixel shuffle 2
        self.conv4 = layers.Conv2D( \
                filters=256, kernel_size=3, strides=1, padding='same')
        self.prelu4 = layers.PReLU(shared_axes=[1, 2])

        # Get three channel output
        self.conv5 = layers.Conv2D( \
                filters=3, kernel_size=3, strides=1, padding='same')

    def call(self, input_tensor, training=False):
        """ Call routine of model. """

        # Pre-residual encoding
        pre_resid = self.conv1(input_tensor)
        pre_resid = self.prelu1(pre_resid)
        x = pre_resid

        # Pass through all resblocks
        for _, resblock in enumerate(self.resblocks):
            x = resblock(x, training=training)

        # Final residual connection block
        x = self.conv2(x)
        x = self.bn2(x)
        x += pre_resid

        # Pixel shuffle 1
        x = self.conv3(x)
        x = tf.nn.depth_to_space(x, block_size=2)
        x = self.prelu3(x)

        # Pixel shuffle 2
        x = self.conv4(x)
        x = tf.nn.depth_to_space(x, block_size=2)
        x = self.prelu4(x)

        # Get three channel output
        x = self.conv5(x)

        return x

    def _init_content_loss_model(self):
        """ Initialized truncated VGG19 model for use in content
        loss function. """

        # Get pretrained VGG19
        vgg = vgg19.VGG19( \
                include_top=False, \
                weights='imagenet')

        for layer in vgg.layers:
            layer.trainable = False

        loss_model = tf.keras.Model(inputs=vgg.input, \
                outputs=vgg.get_layer("block2_conv2").output)
        loss_model.trainable = False

        return loss_model

    def _content_loss(self, output_image, true_image):
        """ Function for calculating the content loss
        of the model (using a pre-trained VGG network)."""

        # Feed model output and true image through the loss model
        preprocessed_output_img = vgg19.preprocess_input(output_image * 225.)
        preprocessed_true_img = vgg19.preprocess_input(true_image * 225.)

        output_features = self.content_loss_model( \
                preprocessed_output_img)
        true_features = self.content_loss_model( \
                preprocessed_true_img)

        # Get Euclidean distance
        loss = tf.reduce_sum(tf.math.square(true_features - output_features)) \
                / (true_features.shape[1] * true_features.shape[2])
        return loss

    def _adversarial_loss(self, disc_output):
        """ Calculates the adversarial loss component of the
        entire loss function of the SRGAN generator network. """

        # return tf.math.reduce_sum(-tf.math.log(disc_output))
        return tf.keras.losses.BinaryCrossentropy(from_logits=False) \
                (tf.ones_like(disc_output), disc_output)

    def loss_fn(self, output_image, true_image, disc_output):
        """ Loss function for the generator network of
        SRGAN. """

        # return self.content_loss(output_image, true_image) \
                # + (1e-3 * self.adversarial_loss(disc_output))
        return tf.reduce_sum(tf.keras.losses.MSE(true_image, output_image)) \
                + (1e-3 * self._adversarial_loss(disc_output)) \
                + (5e-4 * self._content_loss(output_image, true_image))


class GenResBlock(tf.keras.Model):
    """ Defines an SRGAN residual block for the generator
    network. Consists of two convolutional layers as well
    as two batch normalizations, one parametrized ReLU,
    and an element-wise sum of the input tensor at the
    end of the block. """

    def __init__(self):
        super(GenResBlock, self).__init__(name='')

        self.conv1 = layers.Conv2D( \
                filters=64, kernel_size=3, strides=1, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.prelu1 = layers.PReLU(shared_axes=[1, 2])

        self.conv2 = layers.Conv2D( \
                filters=64, kernel_size=3, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        """ Call routine of block. """

        # Convolutional block 1
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = self.prelu1(x)

        # Convolutional block 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # Residual connection
        x += input_tensor

        return x


# ===========================================================================
# DISCRIMINATOR
# ===========================================================================


class SRGAN_Discriminator(tf.keras.Model):
    """ Defines the discriminator network of SRGAN. """

    def __init__(self):
        super(SRGAN_Discriminator, self).__init__(name='')

        # Initial encoding
        self.conv1 = layers.Conv2D( \
                filters=64, kernel_size=3, strides=1, padding='same')
        self.lrelu1 = layers.LeakyReLU(alpha=0.2)

        # Discriminator convolutional blocks
        self.discblocks = [ \
                DiscBlock(64, 2), \
                DiscBlock(128, 1), \
                DiscBlock(128, 2), \
                DiscBlock(256, 1), \
                DiscBlock(256, 2), \
                DiscBlock(512, 1), \
                DiscBlock(512, 2)]

        # Flatten tensor for dense layers
        self.flatten = layers.Flatten()

        # Final dense layers
        self.dense2 = layers.Dense(1024)
        self.lrelu2 = layers.LeakyReLU(alpha=0.2)
        self.dense3 = layers.Dense(1)

    def call(self, input_tensor, training=False):
        """ Call routine of model. """

        # Initial encoding
        x = self.conv1(input_tensor)
        x = self.lrelu1(x)

        # Pass through discriminator convolutional blocks
        for block in self.discblocks:
            x = block(x, training=training)

        # Flatten tensor for dense layers
        x = self.flatten(x)

        # Final dense layers
        x = self.dense2(x)
        x = self.lrelu2(x)
        x = self.dense3(x)
        x = tf.keras.activations.sigmoid(x)

        return x

    def loss_fn(self, real_output, fake_output):
        """ Loss function of discrimnator network of SRGAN. """

        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False) \
                (tf.ones_like(real_output), real_output)
        fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False) \
                (tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss


class DiscBlock(tf.keras.Model):
    """ Defines one convolutional block of the
    discriminator network of SRGAN. """

    def __init__(self, filters, strides):
        super(DiscBlock, self).__init__(name='')

        self.conv = layers.Conv2D( \
                filters=filters, kernel_size=3, \
                strides=strides, padding='same')
        self.bn = layers.BatchNormalization()
        self.lrelu = layers.LeakyReLU(alpha=0.2)

    def call(self, input_tensor, training=False):
        """ Call routine of block. """

        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = self.lrelu(x)

        return x
