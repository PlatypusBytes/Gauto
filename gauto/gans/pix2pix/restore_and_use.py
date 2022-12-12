import tensorflow as tf
import os
from pix2pix_tensorflow_2d import Generator, Discriminator, load_and_normalize_RFs_in_folder, all_images
import plotly.figure_factory as ff
import numpy as np


BATCH_SIZE = 1
test_dataset = load_and_normalize_RFs_in_folder("data\\layers_n\\test\\2d")
input_dataset_test = tf.convert_to_tensor(tf.constant(test_dataset[0]))
test_input_dataset = tf.data.Dataset.from_tensor_slices(input_dataset_test)
test_input_dataset = test_input_dataset.batch(BATCH_SIZE)
target_dataset_test = tf.convert_to_tensor(tf.constant(test_dataset[1]))
test_target_dataset = tf.data.Dataset.from_tensor_slices(target_dataset_test)
test_target_dataset = test_target_dataset.batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.zip((test_input_dataset, test_target_dataset))


OUTPUT_CHANNELS = 1
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
generator=Generator()
discriminator=Discriminator()
checkpoint_dir = "./training_checkpoints_2d/"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=11)
if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    # You can also access previous checkpoints like this: ckpt_manager.checkpoints[3]
    print ('Latest checkpoint restored!!')
    dataset = [value for counter, value in enumerate(test_dataset)]
    predictions = np.array([generator(data[0], training=True) for data in dataset]).flatten()
    target = np.array([data[1] for data in dataset]).flatten()

    # all_images(generator, test_dataset) 
    # Group data together
    hist_data = [target * 4.5 , predictions * 4.5 ]
    group_labels = ['Target data', 'Test results from GAN']
    # Create distplot with custom bin_size
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_rug=False)
    fig.update_layout(template="plotly_white", title_font_family="Times New Roman", font_family="Times New Roman",title={
        'text': "Distribution of IC value"}, font=dict(size=30))
    fig.show()



