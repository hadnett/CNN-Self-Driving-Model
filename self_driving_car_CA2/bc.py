import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras
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
import math


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


def load_steering_img(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
        image_path.append(os.path.join(datadir, left.strip()))
        steering.append(float(indexed_data[3]) + 0.2)
        image_path.append(os.path.join(datadir, right.strip()))
        steering.append(float(indexed_data[3]) - 0.2)
    image_paths = np.asarray(image_path)
    steering = np.asarray(steering)
    return image_paths, steering


# def load_steering_img(datadir, df):
#     image_path = []
#     steering = []
#     for i in range(len(data)):
#         indexed_data = data.iloc[i]
#         center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
#         image_path.append(os.path.join(datadir, center.strip()))
#         steering.append(float(indexed_data[3]))
#         image_path.append(os.path.join(datadir, left.strip()))
#         steering.append(float(indexed_data[3]))
#         image_path.append(os.path.join(datadir, right.strip()))
#         steering.append(float(indexed_data[3]))
#     image_paths = np.asarray(image_path)
#     steering = np.asarray(steering)
#     return image_paths, steering


def random_shadow(img):
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
    img = mpimg.imread(img)
    #Crop the image
    img = img[60:135, :, :]
    # Convert color to yuv y-brightness, u,v chrominants(color)
    # Recommend in the NVIDIA paper
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Apply Gaussian Blur
    # As suggested by NVIDIA paper
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (64, 64))
    img = img/255
    return img


def img_preprocess_no_mread(img):
    #Crop the image
    img = img[60:135, :, :]
    # Convert color to yuv y-brightness, u,v chrominants(color)
    # Recommend in the NVIDIA paper
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Apply Gaussian Blur
    # As suggested by NVIDIA paper
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
    optimizer = Adam(lr=1e-3)
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
    model.add(Dense(1))

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
    zoom_func = iaa.Affine(scale=(1, 1.3))
    z_image = zoom_func.augment_image(image_to_zoom)
    return z_image


def pan(image_to_pan):
    pan_func = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    pan_image = pan_func.augment_image(image_to_pan)
    return pan_image


def img_random_brightness(image_to_brighten):
    bright_func = iaa.Multiply((0.2, 1.2))
    bright_image = bright_func.augment_image(image_to_brighten)
    return bright_image


def img_random_flip(image_to_flip, steering_angle):
    # 0 - flip horizontal, 1 flip vertical, -1 combo of both
    flipped_image = cv2.flip(image_to_flip, 1)
    steering_angle = -steering_angle
    return flipped_image, steering_angle


# def trans_image(image, steer, trans_range):
#     h, w, _ = image.shape
#     tr_x = trans_range * np.random.uniform() - trans_range / 2
#     steer_ang = steer + tr_x / trans_range * 2 * .2
#     tr_y = 40 * np.random.uniform() - 40 / 2
#     # tr_y = 0
#     Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
#     image_tr = cv2.warpAffine(image, Trans_M, (w, h))
#
#     return image_tr, steer_ang


def random_augment(image_to_augment, steering_angle):
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
    return augment_image, steering_angle


def batch_generator(image_paths, steering_ang, batch_size, is_training):
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths)-1)
            if is_training:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]

            im = img_preprocess_no_mread(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))


datadir = '/Volumes/HADNETT/4th_Year/Smart Tech/All_Tracks'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns, encoding='latin-1')
pd.set_option('max_columns', 7)
print(data.head())

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

# num_bins = 23
# avg_samples_per_bin = len(data['steering'])/num_bins
# hist, bins = np.histogram(data['steering'], num_bins)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.plot((np.min(data['steering']), np.max(data['steering'])), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
# plt.show()
#
# print('Total data: ', len(data))
#
# keep_probs = []
# target = avg_samples_per_bin * .5
# for i in range(num_bins):
#     if hist[i] < target:
#         keep_probs.append(1.)
#     else:
#         keep_probs.append(1./(hist[i]/target))
# remove_list = []
# for i in range(len(data['steering'])):
#     for j in range(num_bins):
#         if bins[j] < data['steering'][i] <= bins[j + 1]:
#             # delete from X and y with probability 1 - keep_probs[j]
#             if np.random.rand() > keep_probs[j]:
#                 remove_list.append(i)
#
# print('Remove: ', len(remove_list))
# data.drop(data.index[remove_list], inplace=True)
# print('Remaining: ', len(data))
#
# hist, bins = np.histogram(data['steering'], num_bins)
# plt.bar(center, hist, align='center', width=width)
# plt.plot((np.min(data['steering']), np.max(data['steering'])), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
# plt.show()

num_bins = 25
samples_per_bin = 1100
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

image_paths, steering = load_steering_img(datadir + '/IMG', data)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steering, test_size=0.2, random_state=6)
print('Training samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))



fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title('Training set')
axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
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

#X_train = np.array(list(map(img_preprocess, X_train)))
#X_valid = np.array(list(map(img_preprocess, X_valid)))

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
    original_image = mpimg.imread(random_image)
    augmented_image, steering_angle = random_augment(random_image, random_steering)
    axs[i][0].imshow(original_image)
    axs[i][0].set_title("Original Image")
    axs[i][1].imshow(augmented_image)
    axs[i][1].set_title("Augmented Image")
plt.show()


model = vgg16_model()
print(model.summary())

batch_size = 32
steps_per_epoch_train = int(np.ceil(X_train.shape[0] / batch_size))
steps_per_epoch_val = int(np.ceil(X_valid.shape[0] / batch_size))

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=15)
mc = ModelCheckpoint('best_model_7.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# fit model

h = model.fit(batch_generator(X_train, y_train, batch_size, 1), steps_per_epoch=steps_per_epoch_train*3,
              epochs=3,
              validation_data=batch_generator(X_valid, y_valid, batch_size, 0),
              validation_steps=steps_per_epoch_val*3,
              verbose=1,
              shuffle=1,
              callbacks=[es, mc])

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

model.save('model_19.h5')
