# simple train 
import csv 
import numpy as np 

STEP_SIZE = 0.0001
NUM_STEPS = 1000

# training 
reader = csv.reader(open('/workspaces/CS109/simple-train.csv'))
header = next(reader)
row_count = -1
with open("/workspaces/CS109/simple-train.csv", newline="") as f:
    reader = csv.reader(f) #put all of the training examples into a list 
    x = list(reader) 
    for _ in reader:
        row_count += 1 

num_params = len(header) - 1 #number of params = num_cols - 1
num_exs = row_count #number of training examples = num_rows 

# logistic regression to get parameters 
params = np.zeros(num_params) #initialize parameters to 0 
for i in range(NUM_STEPS):
    gradients = np.zeros(num_params) # initialize gradient to 0 for each iteration 
    for i in range(num_exs):
        for j in range(num_params):
            product = 1 + exp(- params * x[i+1])
            y = x[i+1][-1]
            print(y)
            gradient[j] +=  x[i+1][j] * (y - product)
            print(gradient[j])
            params[j] += gradient[j] * STEP_SIZE
print(params)

# testing 
test_reader = csv.reader(open('/workspaces/CS109/simple-test.csv'))

