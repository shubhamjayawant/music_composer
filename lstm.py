from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint

class LSTM():

    def __init__(self, input_shape, output_shape, weight_dump, weights = None):

        self.model = self.__create_network(input_shape, output_shape)
        self.weight_dump = weight_dump

        if weights:

            self.model.__load_weights(weights)

    def __create_network(self, inputShape, outputShape):

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

    def __load_weights(self, weights):

        self.model.load_weights(weights)

    def train(self, network_input, network_output):

        filepath = self.weight_dump + "weights-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]
        self.model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)