import csv
import cv2
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import transform

from sklearn.model_selection import train_test_split
from random import shuffle

DATA_PATH = './data/data1/'


# sample are line from driving_log.csv.
# the format are [center_img, left_img, right_img, steering]
def _read_file(path):
    samples = []
    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            for i in range(3):
                line[i] = path + line[i].strip()
            samples.append(line[:3] + [float(line[3])])

    # smooth angle
    angles = [line[3] for line in samples]
    for i in range(1, len(angles) - 1):
        samples[i][3] = angles[i-1]/5 + 3*angles[i]/5 + angles[i+1]/5
    return samples


def _random_drop_zero_steering(samples, drop_rate=0.9):
    selected_samples = []

    for line in samples:
        if line[3] != 0:
            selected_samples.append(line)
        elif np.random.uniform(0, 1.) < 1 - drop_rate:
            selected_samples.append(line)
    print('remains %d samples contains steering 0' % np.sum(1 for l in selected_samples if l[3] == 0))
    return selected_samples


def _random_drop_range_steering(samples, lower_bound=-.2, upper_bound=.2, drop_rate=0.5):
    selected_samples = []

    for line in samples:
        if line[3] == 0 or not lower_bound <= line[3] <= upper_bound:
            selected_samples.append(line)
        elif np.random.uniform(0, 1.) < 1 - drop_rate:
            selected_samples.append(line)
    return selected_samples


def get_sample(paths=[DATA_PATH], show_plt=False):
    """
    read samples from data listed in paths. The output can be used in generator to generate feeding data
    :param paths:
    :param show_plt:
    :return:
    """
    samples = []
    for path in paths:
        samples = samples + _read_file(path)
    plt.figure()
    sns.distplot([line[3] for line in samples], kde=False, rug=False, bins=np.arange(-1, 1.01, 0.01))
    plt.title('original steering distribution')
    plt.savefig('img_output/steering_dist.png')
    if show_plt:
        plt.show()

    samples = _random_drop_zero_steering(samples)

    plt.figure()
    sns.distplot([line[3] for line in samples], kde=False, rug=False, bins=np.arange(-1, 1.01, 0.01))
    plt.title('original steering distribution')
    plt.savefig('img_output/steering_dist1.png')
    if show_plt:
        plt.show()

    samples = _random_drop_range_steering(samples)
    plt.figure()
    sns.distplot([line[3] for line in samples], kde=False, rug=False, bins=np.arange(-1, 1.01, 0.01))
    plt.title('original steering distribution')
    plt.savefig('img_output/steering_dist2.png')
    if show_plt:
        plt.show()

    return samples


def _random_select_camera(line, adj_rate=0.2):
    img_idx = np.random.choice([0, 1, 2])
    image_dir = line[img_idx].strip()
    image = cv2.imread(image_dir)
    steering = line[3]
    if img_idx == 1:
        steering += adj_rate
    if img_idx == 2:
        steering -= adj_rate
    return image, steering


def _rotate_img(img, degree):
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def _shift_img(img, x_shift=0, y_shift=0):
    rows, cols = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def trans_image(img, angle, x_trans_limit=60., adj_rate=0.2, y_trans_limit=30, y_adj_mult=0.1):
    """
    randomly shifts image and tune the steering angle
    :param img:
    :param angle:
    :param x_trans_limit:
    :param adj_rate:
    :param y_trans_limit:
    :param y_adj_mult:
    :return:
    """
    # x-axis shift changes angle. we assume 60 pixels make adj_rate shift
    x_shift = np.random.uniform(-x_trans_limit, x_trans_limit)
    img = _shift_img(img, x_shift=x_shift)
    angle += x_shift * adj_rate / x_trans_limit

    ## rotate image changes angle
    # rotage_angle = np.random.uniform(-adj_rate, adj_rate)
    # img = _rotate_img(img, degree=25. * rotage_angle)
    # angle -= rotage_angle

    ## y-axis shift changes angle. we assume 10 pixels make angle * 1.3
    y_shift = np.random.uniform(0, y_trans_limit)
    img = _shift_img(img, y_shift=y_shift)
    angle *= 1 + y_adj_mult * y_shift/y_trans_limit

    return img, angle


def _flip_image(img, steering):
    img = cv2.flip(img, 1)
    steering *= -1.0

    return img, steering


def generator(samples, batch_size=128, adj_rate=0.20):
    """
    generator of data.
    :param batch_size: batch_size
    :param samples: sampled line from driving_log.csv
    :param adj_rate: steering adjust rate
    :return: X_train and y_train
    """
    """
    This generator receives a list of image filenames and steering angles
    and shuffles the data to feed the model.
    Input:
    samples - list from csv file with image filenames and steering angles.
    Arguments:
    batch_size - size of the mini batch
    aug - Data Augmentation flag, if set to True we randomly transform the image.
    adj - steering correction value when using images from the side cameras.
    Output:
    X_train and y_train
    """
    num_samples = len(samples)

    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            image_list = []
            steering_list = []
            for batch_sample in batch_samples:
                image, steering = _random_select_camera(batch_sample, adj_rate=adj_rate)
                image, steering = trans_image(image, steering, adj_rate=adj_rate)

                if np.random.uniform(0., 1.) < 0.50:
                    image, steering = _flip_image(image, steering)
                image_list.append(image)
                steering_list.append(steering)

            X_train = np.array(image_list)
            y_train = np.array(steering_list)

            yield X_train, y_train
