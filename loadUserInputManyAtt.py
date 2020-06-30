import keras
import numpy as np
import time

ask_for_vals ='''Make sure to have spaces in between each entry
Enter the dimensions in inches of your fish for the following:
    Length 1, Length 2, Length 3, Height, Width\n''' 
len_1, len_2, len_3, height, width = input(ask_for_vals).split()
list_vals = [len_1, len_2, len_3, height, width]
empty = []
for number in list_vals:
    empty.append(float(number)*2.54) # change inches to cm (data from csv file in cm)

old_model = keras.models.load_model('fish_dnn_model.h5')
test_data = np.array(empty)
output_pred_grams = old_model.predict(test_data.reshape(1,5), batch_size=1)
output_pred_pounds = output_pred_grams[0]*0.00220462
output_rounded = round(float(output_pred_pounds), 2)
print("Your fish would likely weigh", output_rounded, "pounds\n")
time.sleep(15)
