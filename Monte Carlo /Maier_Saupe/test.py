#!/usr/bin/env python3  
import numpy as np # type: ignore 

# Define the 3 possible states
tensors = np.array([
    [[ 1.0, 0.0, 0.0],
     [ 0.0,-0.5, 0.0],
     [ 0.0, 0.0,-0.5]],

    [[-0.5, 0.0, 0.0],
     [ 0.0, 1.0, 0.0],
     [ 0.0, 0.0,-0.5]],

    [[-0.5, 0.0, 0.0],
     [ 0.0,-0.5, 0.0],
     [ 0.0, 0.0, 1.0]]
]) 


L=20
N = L*L*L
S = 0.0

states = np.random.choice([0,1,2], size=L*L*L)  
print(states)

unique, counts = np.unique(states, return_counts=True)
print(unique, counts)
S = max(counts) / N 
print(S) 

S = (3/2)*S - (1/2)
print(S)



