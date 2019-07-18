import glob
import pickle
import numpy
import config
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import config

class LSTM():

    def __init__(self, inputShape, outputShape, weights = None):

        self.model = self.__createNetwork(inputShape, outputShape)

        if weights:

            self.model.load_weights(weights)

    def __createNetwork(self, inputShape, outputShape):

        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(inputShape[1], inputShape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(outputShape))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        return model

    def loadWeights(self, weights):

        self.model.load_weights(config.PRETRAINED_MODEL)

    def train(self, networkInput, networkOutput):

        filepath = config.WEIGHTS + "weights-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]
        self.model.fit(networkInput, networkOutput, epochs=200, batch_size=64, callbacks=callbacks_list)

# def __load_songs(self):

#     songs = None

#     with open(config.NOTES_ARRAY_DIR + 'songs', 'rb') as filepath:
#         songs = pickle.load(filepath)

#     return songs

def main():

    notes = get_notes()
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)
    
    lstm = LSTM(network_input, n_vocab)
    lstm.train(network_input, network_output)

if __name__ == '__main__':
    
    main()