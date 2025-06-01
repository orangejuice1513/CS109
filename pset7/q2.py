# simple train 
import csv 
import numpy as np 

STEP_SIZE = 0.0001
NUM_STEPS = 1000

# training 
reader = csv.reader(open('data/simple-train.csv'))
header = next(reader)
num_params = len(reader[0]) - 1 #number of params = num_cols - 1
num_exs = len(reader) #number of training examples = num_rows 


# logistic regression to get parameters 
params = np.zeros(num_params) #initialize parameters to 0 
for i in range(NUM_STEPS):
    gradients = np.zeros((num_params) # initialize gradient to 0 for each iteration 
    for i in range(num_exs):
        for j in range(num_params):
            gradient[j] += 
            params[j] += gradient[j] * STEP_SIZE
print(params)

# testing 
test_reader = csv.reader(open('data/simple-test.csv'))