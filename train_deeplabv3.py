# Importing libraries

import pixellib
from pixellib.semantic import semantic_segmentation

from glob import glob
import pandas as pd
from tqdm import tqdm
import shutil
import argparse
import zipfile
import hashlib
import requests
from tqdm import tqdm
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
#from tensorflow.python.keras import Adam
from IPython.display import clear_output
import tensorflow_addons as tfa

AUTOTUNE = tf.data.experimental.AUTOTUNE
print(f"Tensorflow ver. {tf.__version__}")


# Creating variables
# Convert Checkpoint to model
cpk_path = "models/pretrained/deeplabv3_xception65_ade20k_train/" #frozen_inference_graph.pb
root = "gluonCV/"
dataset_path = root + "ADEChallengeData2016/images/"
training_data = "training/"
val_data = "validation/"
# Image size that we are going to use
IMG_SIZE = 128
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Scene Parsing has 150 classes + `not labeled`
N_CLASSES = 151
SEED = 26

# Creating a source dataset

TRAINSET_SIZE = len(glob(dataset_path + training_data + "*.jpg"))
print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

VALSET_SIZE = len(glob(dataset_path + val_data + "*.jpg"))
print(f"The Validation Dataset contains {VALSET_SIZE} images.")

def parse_image(img_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    # For one Image path:
    # .../trainset/images/training/ADE_train_00000001.jpg
    # Its corresponding annotation path is:
    # .../trainset/annotations/training/ADE_train_00000001.png
    mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    # In scene parsing, "not labeled" = 255
    # But it will mess up with our N_CLASS = 150
    # Since 255 means the 255th class
    # Which doesn't exist
    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
    # Note that we have to convert the new value (0)
    # With the same dtype than the tensor itself

    return {'image': image, 'segmentation_mask': mask}

train_dataset = tf.data.Dataset.list_files(dataset_path + training_data + "*.jpg", seed=SEED)
train_dataset = train_dataset.map(parse_image)

val_dataset = tf.data.Dataset.list_files(dataset_path + val_data + "*.jpg", seed=SEED)
val_dataset =val_dataset.map(parse_image)

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

BATCH_SIZE = 2

# for reference about the BUFFER_SIZE in shuffle:
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
BUFFER_SIZE = 1000

dataset = {"train": train_dataset, "val": val_dataset}

# -- Train Dataset --#
dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

#-- Validation Dataset --#
dataset['val'] = dataset['val'].map(load_image_test)
dataset['val'] = dataset['val'].repeat()
dataset['val'] = dataset['val'].batch(BATCH_SIZE)
dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

print(dataset['train'])
print(dataset['val'])

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predicitons
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
        for each pixels.
    """
    # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask
    

model = None

def show_predictions(model=model, dataset=None, num=1):
    """Show a sample prediction.

    Parameters
    ----------
    dataset : [type], optional
        [Input dataset, by default None
    num : int, optional
        Number of sample to show, by default 1
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display_sample([image[0], true_mask, create_mask(pred_mask)])
    else:
        # The model is expecting a tensor of the size
        # [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]
        # but sample_image[0] is [IMG_SIZE, IMG_SIZE, 3]
        # and we want only 1 inference to be faster
        # so we add an additional dimension [1, IMG_SIZE, IMG_SIZE, 3]
        one_img_batch = sample_image[0][tf.newaxis, ...]
        # one_img_batch -> [1, IMG_SIZE, IMG_SIZE, 3]
        inference = model.predict(one_img_batch)
        # inference -> [1, IMG_SIZE, IMG_SIZE, N_CLASS]
        pred_mask = create_mask(inference)
        # pred_mask -> [1, IMG_SIZE, IMG_SIZE, 1]
        display_sample([sample_image[0], sample_mask[0],
                        pred_mask[0]])
        
def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def preprocess_image(image, mask, desired_size=(512, 512)):
    # Resize image and mask
    resized_image = tf.image.resize(image, desired_size)
    resized_mask = tf.image.resize(mask, desired_size, method='nearest')  # Use nearest neighbor for the mask
    return resized_image, resized_mask

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def deeplab_process_batch(images, masks=None, desired_size=(512, 512)):
    # Resize images and masks
    resized_images = tf.image.resize(images, desired_size)
    if masks is not None:
        resized_masks = tf.image.resize(masks, desired_size, method='nearest')
    else:
        resized_masks = None
    return resized_images, resized_masks

def process_batch_wrapper(images, masks):
    resized_images, resized_masks = deeplab_process_batch(images, masks)
    return (resized_images, resized_masks)

# Unbatch the original dataset
dataset_unbatched_train = dataset['train'].unbatch()
dataset_unbatched_val = dataset['val'].unbatch()

# Apply the resizing function to each item in the dataset
dataset_modified = {}
dataset_modified['train'] = dataset_unbatched_train.map(process_batch_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset_modified['val'] = dataset_unbatched_val.map(process_batch_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Rebatch the dataset
dataset_modified['train'] = dataset_modified['train'].batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
dataset_modified['val'] = dataset_modified['val'].batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Inspect the modified dataset structure
for images, masks in dataset_modified['train'].take(1):
    print("Modified shape:", images.shape, masks.shape)

segment_image = semantic_segmentation()
segment_image.load_ade20k_model("models/pretrained/deeplabv3_xception65_ade20k.h5")

deeplabv3plus = segment_image.model2
deeplabv3plus.summary()

model = deeplabv3plus


EPOCHS = 2
LR = 1e-5

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

callbacks = [
    # to collect some useful metrics and visualize them in tensorboard
    tensorboard_callback,
    # if no accuracy improvements we can stop the training directly
    tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
    # to save checkpoints
    tf.keras.callbacks.ModelCheckpoint('best_model_deeplabv3plus_trained.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

#optimizer=tfa.optimizers.RectifiedAdam(lr=1e-3)

optimizer="SGD" #Adam(lr=LR, decay=1e-6) #Adam(learning_rate=LR)

from tensorflow.python.keras.optimizer_v2.adam import Adam
optimizer= Adam(lr=LR, decay=1e-6)

loss = tf.keras.losses.SparseCategoricalCrossentropy()

STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE


model.compile(optimizer=optimizer, loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model_history = model.fit(dataset_modified['train'], epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=dataset_modified['val'],
                          callbacks=callbacks)

# # On CPU
# with tf.device("/cpu:0"):
#     model_history = model.fit(dataset_modified['train'], epochs=EPOCHS,
#                               steps_per_epoch=STEPS_PER_EPOCH,
#                               validation_steps=VALIDATION_STEPS,
#                               validation_data=dataset_modified['val'],
#                               callbacks=callbacks)


def preprocess_image(image, mask, desired_size=(512, 512)):
    # Resize image and mask
    resized_image = tf.image.resize(image, desired_size)
    resized_mask = tf.image.resize(mask, desired_size, method='nearest')  # Use nearest neighbor for the mask
    return resized_image, resized_mask


# Model Evaluation

def deeplab_process_batch(images, masks=None, desired_size=(512, 512)):
    # Resize images and masks
    resized_images = tf.image.resize(images, desired_size)
    if masks!=None:
        resized_masks = tf.image.resize(masks, desired_size, method='nearest')  # Use nearest neighbor for the masks
    else:
        resized_masks = None
    return resized_images, resized_masks


# Metrics
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Accumulate metrics over the validation set
for image_batch, mask_batch in dataset['val'].take(VALIDATION_STEPS):
    image_batch, mask_batch = deeplab_process_batch(image_batch, mask_batch)
    # Make predictions
    predictions = deeplabv3plus.predict(image_batch)

    # Update metrics
    accuracy_metric.update_state(mask_batch, predictions)
    # For mIoU, you need to convert predictions to discrete values (e.g., using argmax)
    predicted_classes = tf.argmax(predictions, axis=-1)

# Get final metric results
accuracy = accuracy_metric.result().numpy()

# Clear the state of the metrics
accuracy_metric.reset_states()

# Print the results
print(f"DeepLabV3Plus Accuracy on Validation Set: {accuracy}")


def deeplab_process_batch(images, masks=None, desired_size=(512, 512)):
    # Resize images and masks
    resized_images = tf.image.resize(images, desired_size)
    if masks is not None:
        resized_masks = tf.image.resize(masks, desired_size, method='nearest')  # Use nearest neighbor for the masks
    else:
        resized_masks = None
    return resized_images, resized_masks

# Initialize dictionary to store accuracy for each class
accuracy_per_class = {f'Class_{i}': tf.keras.metrics.Accuracy() for i in range(NUM_CLASSES)}

# Accumulate metrics over the validation set
for image_batch, mask_batch in tqdm(dataset['val'].take(VALIDATION_STEPS)):
    image_batch, mask_batch = deeplab_process_batch(image_batch, mask_batch)
    # Make predictions
    predictions = deeplabv3plus.predict(image_batch)
    predicted_classes = tf.argmax(predictions, axis=-1)

    # Update per-class accuracy
    for i in range(NUM_CLASSES):
        # Create masks for each class
        true_mask = tf.equal(mask_batch, i)
        pred_mask = tf.equal(predicted_classes, i)

        # Update accuracy for the class
        accuracy_per_class[f'Class_{i}'].update_state(true_mask, pred_mask)

# Extract the final accuracy results for each class
accuracy_results = {class_id: metric.result().numpy() for class_id, metric in accuracy_per_class.items()}

# Clear the state of the metrics
for metric in accuracy_per_class.values():
    metric.reset_states()

# Convert to pandas DataFrame for easy viewing
accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=['Class', 'Accuracy'])

# Print the DataFrame
print(accuracy_df)

accuracy_df.to_excel("DeepLabV3Plus_trained_perClass_acc.xlsx")  