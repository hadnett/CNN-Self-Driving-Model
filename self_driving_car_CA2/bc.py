import os
import pandas as pd
import ntpath
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


def load_driving_data(datadir, df):
    image_path = []
    steering = []
    throttle = []
    reverse = []
    speed = []
    for i in range(len(df)):
        indexed_data = df.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
        throttle.append(float(indexed_data[4]))
        reverse.append(float(indexed_data[5]))
        speed.append(float(indexed_data[6]))
        image_path.append(os.path.join(datadir, left.strip()))
        steering.append(float(indexed_data[3]))
        throttle.append(float(indexed_data[4]))
        reverse.append(float(indexed_data[5]))
        speed.append(float(indexed_data[6]))
        image_path.append(os.path.join(datadir, right.strip()))
        steering.append(float(indexed_data[3]))
        throttle.append(float(indexed_data[4]))
        reverse.append(float(indexed_data[5]))
        speed.append(float(indexed_data[6]))
    image_paths = np.asarray(image_path)
    steering = np.asarray(steering)
    throttle = np.asarray(throttle)
    reverse = np.asarray(reverse)
    speed = np.asarray(speed)
    return image_paths, steering, throttle, reverse, speed


data_dir = '/Volumes/HADNETT/4th_Year/Smart Tech/CA2_Data'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'), names=columns)
pd.set_option('max_columns', 7)
print(data.head())

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

# Steering Data

num_bins = 25
samples_per_bin = 300
hist, bins = np.histogram(data['steering'], num_bins)
print(bins)
center = (bins[:-1] + bins[1:])*0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if bins[j] <= data['steering'][i] <= bins[j + 1]:
            list_.append(i)
    list_ = shuffle(list_, random_state=0)
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

# Throttle Data

num_bins = 5
samples_per_bin = 420
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

# Reverse Data

num_bins = 3
samples_per_bin = 75
hist, bins = np.histogram(data['reverse'], num_bins)
print(bins)
center = (bins[:-1] + bins[1:])*0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['reverse']), np.max(data['reverse'])), (samples_per_bin, samples_per_bin))
plt.show()

remove_list = []
data = data.reset_index(drop=True)
for j in range(num_bins):
    list_ = []
    for i in range(len(data['reverse'])):
        if bins[j] <= data['reverse'][i] <= bins[j + 1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

print('Remove: ', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('Remaining: ', len(data))

hist, _ = np.histogram(data['reverse'], num_bins)
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['reverse']), np.max(data['reverse'])), (samples_per_bin, samples_per_bin))
plt.show()

# Speed Data

num_bins = 10
samples_per_bin = 30
hist, bins = np.histogram(data['speed'], num_bins)
print(bins)
center = (bins[:-1] + bins[1:])*0.5
plt.bar(center, hist)
plt.plot((np.min(data['speed']), np.max(data['speed'])), (samples_per_bin, samples_per_bin))
plt.show()

remove_list = []
data = data.reset_index(drop=True)
for j in range(num_bins):
    list_ = []
    for i in range(len(data['speed'])):
        if bins[j] <= data['speed'][i] <= bins[j + 1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

print('Remove: ', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('Remaining: ', len(data))

hist, _ = np.histogram(data['speed'], num_bins)
plt.bar(center, hist)
plt.plot((np.min(data['speed']), np.max(data['speed'])), (samples_per_bin, samples_per_bin))
plt.show()

image_paths, steering, throttle, reverse, speed = load_driving_data(data_dir + '/IMG', data)
print("Image Paths", image_paths)
print("Steering: ", steering)
print("Throttle: ", throttle)
print("Reverse: ", reverse)
print("Speed: ", speed)
