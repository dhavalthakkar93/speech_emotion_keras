"""keras_training.py: purpose of this script is to train deep neural network using keras on  Toronto emotion speech
dataset to predict the emotion from speech """

__author__ = "Dhaval Thakkar"

import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dropout
from matplotlib import pyplot as plt


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(path):
    features, labels = np.empty((0, 193)), np.empty(0)
    labels = []
    for fn in glob.glob(path):
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        except Exception as e:
            print("Error encountered while parsing file: ", fn)
            continue
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        labels = np.append(labels, fn.split("_")[3].split(".")[0])

    return np.array(features), np.array(labels)


tr_features, tr_labels = parse_audio_files('./training_sounds1/*.wav')


tr_features = np.array(tr_features, dtype=pd.Series)
tr_labels = np.array(tr_labels, dtype=pd.Series)

X = tr_features.astype(int)
Y = tr_labels.astype(str)

seed = 7
numpy.random.seed(seed)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)


def baseline_model():
    deep_model = Sequential()
    deep_model.add(Dense(8, input_dim=193, activation="relu", kernel_initializer="uniform"))
    deep_model.add(Dropout(0.5))
    deep_model.add(Dense(7, activation="relu", kernel_initializer="uniform"))
    deep_model.add(Dropout(0.5))
    deep_model.add(Dense(7, activation="relu", kernel_initializer="uniform"))
    deep_model.add(Dropout(0.5))
    deep_model.add(Dense(7, activation="softmax", kernel_initializer="uniform"))

    deep_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return deep_model


epoches = 1000
batch_size = 25
verbose = 1

model = baseline_model()
result = model.fit(X, dummy_y, validation_split=0.1, batch_size=batch_size, epochs=epoches, verbose=verbose)

# print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

print(result.history)

filename = 'keras_model.h5'

model.save(filename)

print('Model Saved..')

plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('accuracy.png')

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss.png')
