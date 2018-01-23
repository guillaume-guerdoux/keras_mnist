import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras.datasets import mnist

# Load Mnist data
(x_train_basic, y_train_basic), (x_test_basic, y_test_basic) = mnist.load_data()

print(x_train_basic.shape)
# Vision sanity check
plt.imshow(x_train_basic[0])
# plt.show()

# Reshape vector to get only a 28x28 vector
x_train = x_train_basic.reshape(60000, 784)
x_test = x_test_basic.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Tot get value between 0 and 1
x_train /= 255
x_test /= 255

# Pass classes to 10 lenght vectors
y_train = np_utils.to_categorical(y_train_basic, 10)
y_test = np_utils.to_categorical(y_test_basic, 10)

''' 1. Fully connected layer without dropout '''

fc_model = Sequential()
fc_model.add(Dense(512, input_shape=(784,)))
fc_model.add(Activation('relu'))

fc_model.add(Dense(512))
fc_model.add(Activation('relu'))

fc_model.add(Dense(10))
fc_model.add(Activation('softmax'))

fc_model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

fc_model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1)

print(fc_model.metrics_names)
fc_score = fc_model.evaluate(x_test, y_test, verbose=0)
print(fc_score[1])

'''2. Fully connected layer with dropout '''

fc_dropout_model = Sequential()
fc_dropout_model.add(Dense(512, input_shape=(784,)))
fc_dropout_model.add(Activation('relu'))
fc_dropout_model.add(Dropout(0.2))

fc_dropout_model.add(Dense(512))
fc_dropout_model.add(Activation('relu'))
fc_dropout_model.add(Dropout(0.2))

fc_dropout_model.add(Dense(10))
fc_dropout_model.add(Activation('softmax'))

fc_dropout_model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

fc_dropout_model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1)

print(fc_dropout_model.metrics_names)
fc_dropout_score = fc_dropout_model.evaluate(x_test, y_test, verbose=0)
print(fc_dropout_score[1])

'''3. Convolutional NN '''
x_train_conv = x_train_basic.reshape(60000, 28, 28, 1)
x_test_conv = x_test_basic.reshape(x_test_basic.shape[0], 28, 28, 1)

conv_model = Sequential()
conv_model.add(Conv2D(32, (3, 3), activation='relu',
                      input_shape=(28, 28, 1)))
conv_model.add(Conv2D(32, (3, 3), activation='relu'))
conv_model.add(MaxPooling2D(pool_size=(2, 2)))
conv_model.add(Dropout(0.2))

conv_model.add(Flatten())
conv_model.add(Dense(128))
conv_model.add(Activation('relu'))
conv_model.add(Dropout(0.2))

conv_model.add(Dense(10))
conv_model.add(Activation('softmax'))

conv_model.compile(loss='categorical_crossentropy', optimizer='adam',
                   metrics=['accuracy'])

conv_model.fit(x_train_conv, y_train, batch_size=128, epochs=1, verbose=1)

# Save model to json
conv_model_json = conv_model.to_json()
with open("conv_model.json", "w") as json_file:
    json_file.write(conv_model_json)

print(conv_model.metrics_names)
conv_score = conv_model.evaluate(x_test_conv, y_test, verbose=0)
print(conv_score[1])'''

''' Example with conv model '''
'''json_file = open('conv_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

''' Predicition on one example '''
example_digit = x_train_basic[26]
plt.imshow(example_digit)
plt.show()

example_digit = example_digit.reshape(1, 784)

prediction = fc_model.predict(example_digit)
print(np.argmax(prediction))
