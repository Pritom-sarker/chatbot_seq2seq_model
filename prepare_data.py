import tensorflow as tf
import numpy as np
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split

import os
import time

#loaded_model = joblib.load(File_name)
# convert word to number
x=joblib.load('data/input.pkl')
y=joblib.load('data/output.pkl')
char2num=joblib.load('data/char2num.pkl')
num2char=joblib.load('data/num2char.pkl')
# create Vocab

# convert data to numeric value

max_len_x = max([len(date) for date in x])

print(max_len_x)
max_len_y= max([len(date) for date in y])
print(max_len_y)
x = [[char2num['<PAD>']]*(max_len_x - len(date)) +[char2num[x_] for x_ in date] for date in x]
y = [[char2num['<PAD>']]*(max_len_y - len(date)) +[char2num['<GO>']] + [char2num[y_] for y_ in date] for date in y]


print(np.array(y).shape)
joblib.dump(x,"data/final_input.pkl")
joblib.dump(y,"data/final_output.pkl")