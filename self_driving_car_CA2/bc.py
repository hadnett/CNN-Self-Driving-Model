import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


def load_steering_img(datadir, df):
    """
    Loads image paths, steering angles and throttles and stores them in seperate arrays, with an steering angle
    correction of + 0.2 for left images and - 0.2 for right images.
    :param datadir: The data directory of the data
    :param df: The dataframe
    :return: Three arrays containing image paths, steering angle and throttle.
    """
    image_path = []
    steering = []
    throttle = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        throttle.append(float(indexed_data[4]))
        steering.append(float(indexed_data[3]))
        image_path.append(os.path.join(datadir, left.strip()))
        throttle.append(float(indexed_data[4]))
        steering.append(float(indexed_data[3]) + 0.2)
        image_path.append(os.path.join(datadir, right.strip()))
        throttle.append(float(indexed_data[4]))
        steering.append(float(indexed_data[3]) - 0.2)
    image_paths = np.asarray(image_path)
    steering = np.asarray(steering)
    throttle = np.asarray(throttle)
    return image_paths, steering, throttle


def random_shadow(img):
    """
     Augments image by adding a random shadow. https://markku.ai/post/data-augmentation/
     :param img: The img to be augmented
     :return: The augmented image.
     """
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shadow_mask = 0 * hsv[:, :, 1]
    X_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][0]
    Y_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][1]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1

    shadow_density = np.random.uniform(0.4, 0.8)
    left_side = shadow_mask == 1
    right_side = shadow_mask == 0

    if np.random.randint(2) == 1:
        hsv[:, :, 2][left_side] = hsv[:, :, 2][left_side] * shadow_density
    else:
        hsv[:, :, 2][right_side] = hsv[:, :, 2][right_side] * shadow_density

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def img_preprocess(img):
    """
    Preprocesses the images for model training via cropping, adding blur, resizing and normalising.
    :param img: The image to be preprocesses for training.
    :return: The preprocesses image.
    """
    img = mpimg.imread(img)
    #Crop the image
    img = img[60:135, :, :]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (64, 64))
    img = img/255
    return img


def img_preprocess_no_mread(img):
    """
    Preprocesses the images for model training via cropping, adding blur, resizing and normalising without opening
    the image each time.
    :param img: The image to be preprocesses for training.
    :return: The preprocesses image.
    """
    #Crop the image
    img = img[60:135, :, :]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (64, 64))
    img = img/255
    return img


def nvidia_model():
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def nvidia_model_2():
    model = Sequential()
    model.add(Convolution2D(3, 1, 1, padding='same', input_shape=(66, 200, 3), activation='elu'))
    # model.add(Dropout(0.5))

    model.add(Convolution2D(32, 3, 3, padding='same', activation='elu'))
    model.add(Convolution2D(32, 3, 3, padding='same', activation='elu'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.5))

    # In: 32x32
    model.add(Convolution2D(64, 3, 3, padding='same', activation='elu'))
    model.add(Convolution2D(64, 3, 3, padding='same', activation='elu'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.5))

    # In: 16x16
    model.add(Convolution2D(128, 3, 3, padding='same', activation='elu'))
    model.add(Convolution2D(128, 3, 3, padding='same', activation='elu'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(16, activation='elu'))

    model.add(Dense(1, name='output'))
    model.compile(optimizer='Adam', loss='mse')
    return model


def vgg16_model():
    """
    https://neurohive.io/en/popular-networks/vgg16/
    :return: VGG16 Model
    """
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='linear'))

    optimizer = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def simple_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(32, 32, 3)))
    model.add(Convolution2D(15, (3, 3), strides=(2, 2), activation='relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def zoom(image_to_zoom):
    """
    Augment the image by zooming in.
    :param image_to_zoom: The image to zoom
    :return: The zoomed image
    """
    zoom_func = iaa.Affine(scale=(1, 1.3))
    z_image = zoom_func.augment_image(image_to_zoom)
    return z_image


def pan(image_to_pan):
    """
    Pans/translates an image.
    :param image_to_pan: The image to be panned
    :return: The panned image
    """
    pan_func = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    pan_image = pan_func.augment_image(image_to_pan)
    return pan_image


def img_random_brightness(image_to_brighten):
    """
    Adds random brightness to an image between ranges of 0.2 and 1.2.
    :param image_to_brighten: The image to be brightened
    :return: The brightened image
    """
    bright_func = iaa.Multiply((0.2, 1.2))
    bright_image = bright_func.augment_image(image_to_brighten)
    return bright_image


def img_random_flip(image_to_flip, steering_angle):
    """
    Flips an image, either horizontally, vertically or both.
    :param image_to_flip: The image to flip
    :param steering_angle: The steering angle to be adjusted after flip
    :return: The image flipped and the steering angle adjusted.
    """
    # 0 - flip horizontal, 1 flip vertical, -1 combo of both
    flipped_image = cv2.flip(image_to_flip, 1)
    steering_angle = -steering_angle
    return flipped_image, steering_angle


def random_augment(image_to_augment, steering_angle, throttle):
    """
    Randomly applies augmentation functions to images.
    :param image_to_augment: The image to be augmented
    :param steering_angle: The steering angle to be adjusted as a result of augmentation.
    :param throttle: The throttle to be adjusted as a result of augmentation.
    :return: The randomly augmented image, steering angle and throttle
    """
    augment_image = mpimg.imread(image_to_augment)
    if np.random.rand() < 0.5:
        augment_image = zoom(augment_image)
    if np.random.rand() < 0.5:
        augment_image = pan(augment_image)
    if np.random.rand() < 0.5:
        augment_image = img_random_brightness(augment_image)
    if np.random.rand() < 0.5:
        augment_image, steering_angle = img_random_flip(augment_image, steering_angle)
    if np.random.rand() < 0.5:
        augment_image = random_shadow(augment_image)
    return augment_image, steering_angle, throttle


def batch_generator(image_paths, steering_ang, batch_size, is_training):
    """
    Creates batches of images to be passed to the deep learning NN.
    :param image_paths: The image paths for training.
    :param steering_ang: The steering angle for training.
    :param batch_size: The size of the batch being used for training.
    :param is_training: Whether the model is training or not.
    :return: A batch of data containing images paths, steering angles and throttle for model training.
    """
    while True:
        batch_img = []
        batch_steering = []
        batch_throttle = []
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths)-1)
            if is_training:
                im, steering, throttle = random_augment(image_paths[random_index], steering_ang.iloc[random_index, 0]
                                                        , steering_ang.iloc[random_index, 1])
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang.iloc[random_index, 0]
                throttle = steering_ang.iloc[random_index, 1]

            im = img_preprocess_no_mread(im)
            batch_img.append(im)
            batch_steering.append(steering)
            batch_throttle.append(throttle)
            combined = np.vstack((batch_steering, batch_throttle)).T
        yield (np.asarray(batch_img), combined)


datadir = '/Volumes/HADNETT/4th_Year/Smart Tech/All_Tracks'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns, encoding='latin-1')
pd.set_option('max_columns', 7)
print(data.head())

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

data = shuffle(data)
data = data.reset_index(drop=True)

num_bins = 23
samples_per_bin = 1020
hist, bins = np.histogram(data['steering'], num_bins)
print(bins)
center = (bins[:-1] + bins[1:])*0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

print('Total data: ', len(data))

remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if bins[j] <= data['steering'][i] <= bins[j + 1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

print('Remove: ', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('Remaining: ', len(data))

hist, _ = np.histogram(data['steering'], num_bins)
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

print(data.iloc[1])

num_bins = 5
samples_per_bin = 2600
hist, bins = np.histogram(data['throttle'], num_bins)
print(bins)
center = (bins[:-1] + bins[1:])*0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['throttle']), np.max(data['throttle'])), (samples_per_bin, samples_per_bin))
plt.show()

# Indexes where changed after previous delete. Therefore, they needed to be reset in order to preform
# an additional delete. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html
remove_list = []
data = data.reset_index(drop=True)
for j in range(num_bins):
    list_ = []
    for i in range(len(data['throttle'])):
        if bins[j] <= data['throttle'][i] <= bins[j + 1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

print('Remove: ', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('Remaining: ', len(data))

hist, _ = np.histogram(data['throttle'], num_bins)
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['throttle']), np.max(data['throttle'])), (samples_per_bin, samples_per_bin))
plt.show()

print(data.iloc[1])

image_paths, steering, throttle = load_steering_img(datadir + '/IMG', data)

combined_data = pd.DataFrame({'steering': steering, 'throttle': throttle})
print("Combined Data: ", combined_data.head(10))

X_train, X_valid, y_train, y_valid = train_test_split(image_paths, combined_data, test_size=0.2, random_state=6)
print('Training samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

num_bins = 23
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train['steering'], bins=num_bins, width=0.05, color='b')
axes[0].set_title('Training set')
axes[1].hist(y_valid['steering'], bins=num_bins, width=0.05, color='r')
axes[1].set_title('Validation set')
plt.show()

image = image_paths[100]
original_image = mpimg.imread(image)
preprocessed_image = img_preprocess(image)
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[1].imshow(preprocessed_image)
axes[1].set_title('Preprocessed Image')
plt.show()

x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(x_train_gen[0])
axs[0].set_title("Training Image")
axs[1].imshow(x_valid_gen[0])
axs[1].set_title("Validation Image")
plt.show()

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[1].imshow(zoomed_image)
axs[1].set_title("Zoomed Image")
plt.show()

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[1].imshow(panned_image)
axs[1].set_title("Panned Image")
plt.show()

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
bright_image = img_random_brightness(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[1].imshow(bright_image)
axs[1].set_title("Bright Image")
plt.show()

random_index = random.randint(0, 1000)
image = image_paths[random_index]
steering_angle = steering[random_index]
original_image = mpimg.imread(image)
flipped_image, flipped_angle = img_random_flip(original_image, steering_angle)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image - " + "Steering Angle: " + str(steering_angle))
axs[1].imshow(flipped_image)
axs[1].set_title("Flipped Image"+ "Steering Angle: " + str(flipped_angle))
plt.show()

ncols = 2
nrows = 10
fig, axs = plt.subplots(nrows, ncols, figsize=(15, 50))
fig.tight_layout()
for i in range(10):
    rand_num = random.randint(0, len(image_paths)-1)
    random_image = image_paths[rand_num]
    random_steering = steering[rand_num]
    random_throttle = throttle[rand_num]
    original_image = mpimg.imread(random_image)
    augmented_image, steering_angle, throttle = random_augment(random_image, random_steering, throttle)
    axs[i][0].imshow(original_image)
    axs[i][0].set_title("Original Image")
    axs[i][1].imshow(augmented_image)
    axs[i][1].set_title("Augmented Image")
plt.show()

model = vgg16_model()
print(model.summary())

batch_size = 32
# Steps per Epoch equation: https://stackoverflow.com/a/49924566
steps_per_epoch_train = int(np.ceil(X_train.shape[0] / batch_size))
steps_per_epoch_val = int(np.ceil(X_valid.shape[0] / batch_size))


es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=3)
mc = ModelCheckpoint('best_model_22.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

h = model.fit(batch_generator(X_train, y_train, batch_size, 1), steps_per_epoch=steps_per_epoch_train*8,
              epochs=18,
              validation_data=batch_generator(X_valid, y_valid, batch_size, 0),
              validation_steps=steps_per_epoch_val*8,
              verbose=1,
              shuffle=1,
              callbacks=[es, mc])

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

model.save('model_33.h5')
