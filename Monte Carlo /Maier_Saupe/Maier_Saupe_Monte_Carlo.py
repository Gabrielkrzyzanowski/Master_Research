#!/usr/bin/env python3    
"""
Monte Carlo simulation of the Maier Saupe Model
"""
import numpy as np  # type: ignore


"""
n=5
spins = np.random.choice([1,2,3,4,5], size=(n,n))   
flat_spins = np.ravel(spins) 

print(f'spins:\n{spins}')
print(f'flat_spins:\n{flat_spins}') 


i = 0
print(f'-------------- Neighbours\nflat_spin[i] = {flat_spins[i]}')
print(f'spins[i+1,j] = flat_spins[i + n] = {flat_spins[((i // n + 1) % n) * n + (i % n)]}')
print(f'spins[i-1,j] = flat_spins[i - n] = {flat_spins[((i // n - 1) % n) * n + (i % n)]}')
print(f'spins[i,j+1] = flat_spins[i + 1] = {flat_spins[(i // n) * n + ((i % n + 1) % n)]}')
print(f'spins[i,j-1] = flat_spins[i - 1] = {flat_spins[(i // n) * n + ((i % n - 1) % n)]}') 


""" 

def metropolis_sweepV2(spins, probs, n):

    for i in range(len(spins)-1): 
        de = 2 * spins[i] * ( 
            spins[((i // n + 1) % n) * n + (i % n)] + 
            spins[((i // n - 1) % n) * n + (i % n)] + 
            spins[(i // n) * n + ((i % n + 1) % n)] + 
            spins[(i // n) * n + ((i % n - 1) % n)] 
        )
        if np.random.random() < probs[de + 8]:
            spins[i] *= -1

def metropolis_sweepV1(spins, length, probs): 

    xm = length - 2
    x = length - 1
    for xp in range(length):
        ym = length - 2
        y = length - 1
        for yp in range(length):
            de = 2 * spins[x, y] * (spins[xp, y] + spins[xm, y] + spins[x, yp] + spins[x, ym])
            if np.random.random() < probs[de + 8]:
                spins[x, y] *= -1
            ym = y
            y = yp
        xm = x
        x = xp


def set_tabs(beta):
    prob = np.zeros(17)
    for i in range(-8, 9):
        prob[i + 8] = np.exp(-beta*i) if i > 0 else 1  
    
    return prob

n=100
#spins = np.random.choice([-1,1], size=(n,n))   
spins = np.ones((1, n*n), dtype=int) 
flat_spins = np.ravel(spins) 


probs= set_tabs(1/2.0) 

for _ in range(50):
    #metropolis_sweepV1(spins, n, probs)
    metropolis_sweepV2(spins, n, probs)

