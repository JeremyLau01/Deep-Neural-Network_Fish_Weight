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

### Adding three layer of neurons
# More neurons and layers --> slower to train network
# Create first layer with 5 neurons
# Relu activation function normalizes all values and puts them on a scale 0 to 1
# Relu considered to be best activation fxn in many cases
# input_shape is shape of X (the inputs to the first layer are the X data)
model.add(keras.layers.Dense(5, activation='relu', input_shape=(5,)))
# Create second layer with 5 neurons
# After the first layer, don't need to specify the size of the input to the layer
'''can add any number of hidden layers, but 1 or 2 should always work well'''
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
# 5 in EarlyStopping specifies # of pochs with no improvement after which training will be stopped
model.fit(X, Y, epochs=90, batch_size=32 ,callbacks=[keras.callbacks.EarlyStopping(patience=5)])

### Testing model on first datapoint b/c we know the true values - lets see how our model does
# generally you want to test model on new data
test_data = np.array([23.2, 25.4, 30, 11.52, 4.02])
# printing out the output when the test data is run through the model
# need to reshape it b/c it is a row, not a column
# only 1 batch
print(model.predict(test_data.reshape(1,5), batch_size=1))

### Saving model, can load it later in other programs
model.save('fish_dnn_model.h5')

### If want to use the saved model on new data (loading same weights and biases)
old_model = keras.models.load_model('fish_dnn_model.h5')
test_data = np.array([23.2, 25.4, 30, 11.52, 4.02])
print(old_model.predict(test_data.reshape(1,5), batch_size=1))
