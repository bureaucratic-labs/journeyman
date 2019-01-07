import json
import pickle

from sklearn import preprocessing

from keras import Sequential
from keras.models import load_model as load_keras_model
from keras.layers import Bidirectional, Embedding, GRU, Dense
from keras.preprocessing.sequence import pad_sequences


class IndexTransformer:

    def __init__(self):
        self.alphabet = set()
        self.index = {}

    def fit(self, train_x):
        self.alphabet = set(
            sum([[char for char in text] for text in train_x], [])
        )
        for i, char in enumerate(self.alphabet):
            self.index[char] = i + 1
        return self

    def transform(self, X, maxlen):
        return pad_sequences([
            [self.index.get(char, 0) for char in text] for text in X
        ], maxlen=maxlen)

    def __len__(self):
        return len(self.index) + 1


class Sequence:

    def __init__(self, maxlen, embedding_dim=128, units=128, dropout=0.5, batch_size=32, preprocessor=None, labels=None):
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.units = units
        self.dropout = dropout
        self.batch_size = batch_size
        self.index = preprocessor or IndexTransformer()
        self.lb = preprocessing.LabelBinarizer()
        self.model = None
        if labels:
            self.lb.fit(labels)

    def get_model(self):
        model = Sequential()
        model.add(Embedding(len(self.index), self.embedding_dim))
        model.add(
            Bidirectional(GRU(self.units, dropout=self.dropout, recurrent_dropout=self.dropout))
        )
        model.add(Dense(self.maxlen * 2))
        model.add(Dense(len(self.lb.classes_), activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )
        return model

    def fit(self, train_x, train_y, epochs):
        labels = self.lb.fit_transform(train_y)
        train_data = self.index.fit(train_x).transform(train_x, maxlen=self.maxlen)
        self.model = self.get_model()
        self.model.fit(train_data, labels, epochs=epochs, batch_size=self.batch_size)

    def score(self, test_x, test_y):
        test_y = self.lb.transform(test_x)
        test_x = self.index.transform(test_x, maxlen=self.maxlen)
        return self.model.evaluate(test_x, test_y, batch_size=self.batch_size)

    def predict(self, X):
        X = self.index.transform(X, maxlen=self.maxlen)
        Y = self.model.predict(X, batch_size=self.batch_size)
        return self.lb.inverse_transform(Y)

    def save(self, model_path, preprocessor_path, params_path):
        self.model.save(model_path)
        with open(preprocessor_path, 'wb') as output:
            output.write(pickle.dumps(self.index))
        with open(params_path, 'w') as output:
            output.write(json.dumps({
                'maxlen': self.maxlen,
                'embedding_dim': self.embedding_dim,
                'units': self.units,
                'dropout': self.dropout,
                'batch_size': self.batch_size,
                'labels': list(self.lb.classes_),
            }, indent=4))


def load_data_and_labels(path):
    x, y = [], []
    with open(path) as source:
        for line in source:
            if not line:
                continue
            try:
                x_, y_ = line.rsplit('\t', maxsplit=1)
                x.append(x_.strip())
                y.append(y_.strip())
            except ValueError:
                pass
    return x, y


def load_model(model_path, preprocessor_path, params_path):
    params = None
    preprocessor = None
    with open(params_path) as source:
        params = json.loads(source.read())
    with open(preprocessor_path, 'rb') as source:
        preprocessor = pickle.loads(source.read())
    model = Sequence(preprocessor=preprocessor, **params)
    model.model = load_keras_model(model_path)
    return model
