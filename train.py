from matplotlib import pyplot as plt
import tensorflow as tf
import preprocess

def train_GAN(gen_model, disc_model, data):

    # Images used in this epoch, selected at random
    img_ds = preprocess.get_image_set(data, 400)

    train_ds = preprocess.get_batches(img_ds, 32)

    for i, (input_batch, target_batch) in enumerate(train_ds):

        # Train on batch of crops
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # Generate fake image
            fake_batch = gen_model(input_batch, training=True)

            # Discriminate fake image
            fake_discrim = disc_model(fake_batch, training=True)

            # Discriminate true image
            true_discrim = disc_model(target_batch, training=True)

            # Calculate losses
            gen_loss = gen_model.loss_fn(fake_batch, target_batch, fake_discrim)
            disc_loss = disc_model.loss_fn(true_discrim, fake_discrim)

        # Calculate gradients
        gen_grads = gen_tape.gradient( \
                gen_loss, gen_model.trainable_variables)
        disc_grads = disc_tape.gradient( \
                disc_loss, disc_model.trainable_variables)

        # Apply gradients
        gen_model.optimizer.apply_gradients( \
                zip(gen_grads, gen_model.trainable_variables))
        disc_model.optimizer.apply_gradients( \
                zip(disc_grads, disc_model.trainable_variables))

        gen_progress = "Generator loss {0:.4f}".format(gen_loss)
        disc_progress = "Discriminator loss {0:.4f}".format(disc_loss)

        # Print progress
        print("Batch {}:".format(i+1))
        print("|----> " + gen_progress)
        print("|----> " + disc_progress)
        print()
