import keras
import numpy as np

# Ask user for inputs
ask_for_vals ='''Make sure to have spaces in between each entry
Enter the dimensions in inches of your fish for the following:
    Length 1, Length 2, Length 3, Height, Width\n''' 

# Convert user inputs to list which will input into saved model from previous python script
len_1, len_2, len_3, height, width = input(ask_for_vals).split()
list_vals = [len_1, len_2, len_3, height, width]
formatted_inputs = []
for number in list_vals:
    formatted_inputs.append(float(number)*2.54) # change inches to cm (data from csv file in cm)

# Load old model
old_model = keras.models.load_model('fish_dnn_model.h5')
test_data = np.array(formatted_inputs) # converting user inputted values to an array
output_pred_grams = old_model.predict(test_data.reshape(1,5), batch_size=1) # run data through the loaded model
output_pred_pounds = output_pred_grams[0]*0.00220462 # change grams to pounds
output_rounded = round(float(output_pred_pounds), 2) # round to two decimal places

# Display estimated fish weight
print("Your fish would likely weigh", output_rounded, "pounds\n")
