import keras
import numpy as np
import pandas as pd

df = pd.read_csv('fish_data.csv')
#use print(df.head()) to see first five values of dataset to see what data looks like

### Setting x and y values
# X is everything but the weight and species columns
X = df.drop(columns=['Weight','Species'])
# Y is the weight column
Y = df[['Weight']]

### Building model
model = keras.Sequential()

### Adding four layer of neurons (1 input layer, 2 hidden layers, 1 output layer)
# More neurons and layers --> slower to train network
# Create first layer with 5 neurons
# Relu activation function normalizes all values and puts them on a scale 0 to 1
# Relu considered to be best activation fxn in many cases
# input_shape is shape of X (the inputs to the first layer are the X data)
model.add(keras.layers.Dense(5, activation='relu', input_shape=(5,)))
# Create second layer with 6 neurons
# After the first layer, don't need to specify the size of the input to the layer
model.add(keras.layers.Dense(6, activation='relu'))
# Create third layer with 5 neurons
model.add(keras.layers.Dense(5, activation='relu'))
# Create last layer, outputting one value to one neuron (outputting fish weight)
model.add(keras.layers.Dense(1))

### Compile model (specifing optimizer and loss function)
# Adam is considered to be best optimizer and using mse loss fxn
model.compile(keras.optimizers.Adam(lr=0.1), loss='mean_squared_error')
#optimizer='adam' doesn't work as well for some reason

### Fitting model (aka training the model)
# epochs means that going to go over data 30 times
# if model's accuracy not improving by much after iterations, then
# callbacks will stop the training early even if haven't gone through all epochs
# 5 in EarlyStopping specifies # of epochs with no improvement after which training will be stopped
model.fit(X, Y, epochs=90, batch_size=32 ,callbacks=[keras.callbacks.EarlyStopping(patience=5)])

### Testing model on first datapoint b/c we know the true values - lets see how our model does
# generally you want to test model on new data
test_data = np.array([23.2, 25.4, 30, 11.52, 4.02])
# printing out the output when the test data is run through the model
# reshape test data from row to a column, this is only one batch
print(model.predict(test_data.reshape(1,5), batch_size=1))

### Saving model, can load it later in next script
model.save('fish_dnn_model.h5')
