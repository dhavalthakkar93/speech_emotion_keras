import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dropout

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
    model = Sequential()
    """model.add(Dense(8, input_dim=193, activation='relu'))
    model.add(Dense(7, activation='softmax'))"""
    model.add(Dense(8, input_dim=193, activation="relu", kernel_initializer="uniform"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="relu", kernel_initializer="uniform"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax", kernel_initializer="uniform"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


epoches = 500
batch_size = 5
verbose = 1

estimator = KerasClassifier(build_fn=baseline_model, epochs=epoches, batch_size=batch_size, verbose=verbose)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
model = baseline_model()
model.fit(X, dummy_y, batch_size=batch_size, epochs=epoches, verbose=verbose)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

filename = 'keras_model.h5'

model.save(filename)

print('Model Saved..')

