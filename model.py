import tensorflow as tf
from tensorflow.keras import layers

class SRGAN_Generator(tf.keras.Model):
    """ Defines the generator network of SRGAN. """

    def __init__(self, B):
        super(SRGAN_Generator, self).__init__(name='')

        # Pre-residual encoding
        self.conv1 = layers.Conv2D( \
                filters=64, kernel_size=9, strides=1, padding='same')
        self.prelu1 = layers.PReLU()

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
        self.prelu3 = layers.PReLU()

        # Pixel shuffle 2
        self.conv4 = layers.Conv2D( \
                filters=256, kernel_size=3, strides=1, padding='same')
        self.prelu4 = layers.PReLU()

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
        self.prelu1 = layers.PReLU()

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

        # Final dense layers
        x = self.dense2(x)
        x = self.lrelu2(x)
        x = self.dense3(x)
        x = tf.keras.activations.sigmoid(x)

        return x


class DiscBlock(tf.keras.Model):
    """ Defines one convolutional block of the
    discriminator network of SRGAN. """

    def __init__(self, filters, strides):
        super(DiscBlock, self).__init__(name='')

        self.conv = layers.Conv2D( \
                filters=filters, kernel_size=3, strides=strides, padding='same')
        self.bn = layers.BatchNormalization()
        self.lrelu = layers.LeakyReLU(alpha=0.2)

    def call(self, input_tensor, training=False):
        """ Call routine of block. """

        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = self.lrelu(x)

        return x
