import math

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Cropping2D, Convolution2D
from keras.layers import SpatialDropout2D, Lambda, Dropout, Activation
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger

from sklearn.model_selection import train_test_split
from utils import get_sample, generator

DATA_PATHS = ['./data/data/', './data/data1/']
CHECKPOINT_DIR = './nvidia_model_checkpoints/model_{epoch:02d}.h5'
LOG_DIR = './nvidia_model_checkpoints/train_log.csv'
REGULARIZER = 1e-3
ADJUST_RATE = 0.20
BATCH_SIZE = 128


def nvidia_model(input_shape):
    """
    model referenced from
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    I modified the model by adding Convolution2D and W_regularizer, and change activation function to elu
    """
    _model = Sequential()
    _model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
    _model.add(Lambda(lambda x: x / 255. - 0.5))
    _model.add(Convolution2D(24, (5, 5), padding="same", strides=(2, 2), activation="elu"))
    _model.add(SpatialDropout2D(0.2))
    _model.add(Convolution2D(36, (5, 5), padding="same", strides=(2, 2), activation="elu"))
    _model.add(SpatialDropout2D(0.2))
    _model.add(Convolution2D(48, (5, 5), padding="valid", strides=(2, 2), activation="elu"))
    _model.add(SpatialDropout2D(0.2))
    _model.add(Convolution2D(64, (3, 3), padding="valid", strides=(2, 2), activation="elu"))
    _model.add(SpatialDropout2D(0.2))
    _model.add(Convolution2D(64, (3, 3), padding="valid", strides=(2, 2), activation="elu"))
    _model.add(SpatialDropout2D(0.2))

    _model.add(Flatten())
    _model.add(Dropout(0.2))
    _model.add(Dense(100, activation="elu", kernel_regularizer=l2(REGULARIZER)))
    _model.add(Dropout(0.2))
    _model.add(Dense(50, activation="elu", kernel_regularizer=l2(REGULARIZER)))
    _model.add(Dropout(0.2))
    _model.add(Dense(10, activation="elu", kernel_regularizer=l2(REGULARIZER)))
    _model.add(Dropout(0.2))
    _model.add(Dense(1))

    _model.compile(optimizer=Adam(lr=.001), loss='mse')
    return _model


if __name__ == '__main__':
    last_training_checkpoint = 0  # int

    samples = get_sample(DATA_PATHS)
    # training_samples, val_samples = train_test_split(samples, test_size=0.2)
    training_samples = samples
    val_samples = samples

    model = nvidia_model(input_shape=(160, 320, 3))
    checkpoint = ModelCheckpoint(CHECKPOINT_DIR)
    csv_logger = CSVLogger(LOG_DIR, append=last_training_checkpoint>0)

    if last_training_checkpoint > 0:
        model = load_model(CHECKPOINT_DIR.format(epoch=last_training_checkpoint))
        model.compile(optimizer=Adam(lr=.001), loss='mse')

    model.summary()
    model.fit_generator(generator(training_samples, batch_size=BATCH_SIZE, adj_rate=ADJUST_RATE),
                        epochs=50,
                        steps_per_epoch=math.ceil(len(training_samples) / BATCH_SIZE)*10,
                        validation_data=generator(val_samples, batch_size=BATCH_SIZE, adj_rate=ADJUST_RATE),
                        validation_steps=math.ceil(len(val_samples) / BATCH_SIZE),
                        callbacks=[checkpoint, csv_logger],
                        verbose=1,
                        initial_epoch=last_training_checkpoint)

    model.save('model.h5')
