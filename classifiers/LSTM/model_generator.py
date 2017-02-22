from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import recurrent
import numpy as np
import utils.dataprep as dp
from keras.utils.np_utils import to_categorical
from keras.models import load_model

def generateModel(X_train,Y_train,X_test,Y_test):
    model = Sequential()
    model.add(recurrent.LSTM(32, input_dim=1, input_length=99, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train = np.array(X_train).reshape(-1,99,1)
    X_test = np.array(X_test).reshape(-1,99,1)

    Y_train = to_categorical(Y_train, 10)
    Y_test = to_categorical(Y_test, 10)

    # Fit the model
    model.fit(X_train, Y_train, nb_epoch=150, batch_size=4)

    # model.predict(X_test, batch_size=4, verbose=0)
    # model.predict_on_batch(self, x)

    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    # del model  # deletes the existing model

    # returns a compiled model
    # identical to the previous one
    # model = load_model('my_model.h5')

    scores = model.evaluate(X_test, Y_test, batch_size=4)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    return model