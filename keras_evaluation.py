import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import accuracy_score


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


target_files = []


def parse_audio_files(path):
    labels = []
    features = np.empty((0, 193))
    for fn in glob.glob(path):
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        except Exception as e:
            print("Error encountered while parsing file: ", fn)
            continue
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        labels = np.append(labels, fn.split("_")[3].split(".")[0])
        target_files.append(fn)
    return np.array(features), np.array(labels)


ts_features, ts_labels = parse_audio_files('./test_sounds/*.wav')

ts_features = np.array(ts_features, dtype=pd.Series)
ts_labels = np.array(ts_labels, dtype=pd.Series)

test_true = ts_labels
test_class_label = ts_labels

encoder = LabelEncoder()
encoder.fit(ts_labels.astype(str))
encoded_Y = encoder.transform(ts_labels.astype(str))

ts_labels = np_utils.to_categorical(encoded_Y)

ts_labels.resize(ts_labels.shape[0], 7)

filename = 'keras_model.sav'

model = load_model('keras_model.h5')

prediction = model.predict_classes(ts_features.astype(int))


test_predicted = []

labels_map = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

for i, val in enumerate(prediction):
    test_predicted.append(labels_map[val])

print(test_predicted)
print("Accuracy Score:", accuracy_score(test_true, test_predicted))
print('Number of correct prediction:', accuracy_score(test_true, test_predicted, normalize=False), 'out of', len(ts_labels))

matrix = confusion_matrix(test_true, test_predicted)
classes = list(set(test_class_label))
classes.sort()
df = pd.DataFrame(matrix, columns=classes, index=classes)
plt.figure()
sn.heatmap(df, annot=True)
plt.show()

