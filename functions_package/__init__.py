import tensorflow as tf
import numpy as np

# set fixed seed to ensure the same randomness of weight initialization
import random as rn 
import os 
os.environ['PYTHONHASHSEED'] = '0' 
np.random.seed(70) 
rn.seed(70) 
tf.set_random_seed(70) 

# configure session
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
session = tf.Session(config=config) 


