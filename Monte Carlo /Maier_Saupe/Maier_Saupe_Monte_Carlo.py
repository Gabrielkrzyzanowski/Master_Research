#!/usr/bin/env python3    

"""
Monte Carlo simulation of the 3D Maier-Saupe model using the Metropolis algorithm.

This class simulates liquid crystal ordering on a cubic lattice of size 
(length * length * length), where each lattice site can occupy one of three 
discrete orientations. The simulation computes the nematic order parameter `S` 
as a function of temperature, averaged over multiple independent runs.

Usage
----------
Run the script from the command line with the following arguments:

    python script_name.py length t_min t_max t_step

where:
length : int
    Linear size of the cubic lattice (system size = length^3).
t_min : float
    Minimum simulation temperature.
t_max : float
    Maximum simulation temperature.
t_step : float
    Step size for temperature sweep.

Notes
-----
- The Metropolis algorithm is used for spin updates with periodic boundary 
    conditions in 3D.
- The order parameter `S` is computed as:
        S = (3/2) * (fraction in majority state) - 1/2
- Results are saved in `Data/MaierSaupe_MonteCarlo_L{length}.csv`
    with columns: Temperature, S average, S error.
    
"""
import numpy as np  # type: ignore
import time
import argparse 
from concurrent.futures import ProcessPoolExecutor  

def set_tabs(beta):
    """
    Precomputes the Metropolis transition probabilities

    Parameters
    ---------- 
    beta : float
        Experiment temperature. 

    Returns
    -------
    prob : 
        1D array representing the transition probabilities.
    """
    prob = np.zeros(13)
    for i in range(-6, 7):
        prob[i + 6] = np.exp(-beta*i) if i > 0 else 1  

    return prob

def metropolis_sweep(states, tensors, probs, n):
    """
    Performs one Monte Carlo sweep using the Metropolis algorithm
    """ 
    for i in range(len(states)-1): 

        #Find the neighbour's index
        x = i // (n*n)
        y = (i // n) % n
        z = i % n
        v1 = states[(( (x+1) % n ) * n*n) + (y*n) + z]
        v2 = states[(( (x-1) % n ) * n*n) + (y*n) + z]
        v3 = states[(x*n*n) + (( (y+1) % n )*n) + z]
        v4 = states[(x*n*n) + (( (y-1) % n )*n) + z]
        v5 = states[(x*n*n) + (y*n) + ((z+1) % n)]
        v6 = states[(x*n*n) + (y*n) + ((z-1) % n)]

        #Calculate the two states flip's energy difference 
        neigh_sum = tensors[v1] + tensors[v2] + tensors[v3] + tensors[v4] + tensors[v5] + tensors[v6]
        de1 = (4/9)*(np.trace(( - tensors[ (states[i] + 1 ) % 3 ] + tensors[states[i]] ) @ neigh_sum ))
        de2 = (4/9)*(np.trace(( - tensors[ (states[i] + 2 ) % 3 ] + tensors[states[i]] ) @ neigh_sum ))
        
        if min(de1,de2) <= 0: 
            states[i] = (states[i] + 1) % 3 if de1<de2 else (states[i] + 2) % 3
        else:
            if np.random.random() < probs[int(min(de1,de2) + 6)]:
                states[i] = (states[i] + 1) % 3 if de1<de2 else (states[i] + 2) % 3

def parallel_run(equilibrium_time, measurement_time, states, tensors, probs, n):
    """
    Run a Monte Carlo simulation to compute the order parameter.

    Parameters
    ----------
    equilibrium_time : int
        Number of sweeps to equilibrate the system.
    measurement_time : int
        Number of sweeps over which measurements are taken.
    states : ndarray
        3D array representing the lattice configuration.
    tensors : ndarray 
        2D array representing the possible directions
    probs : dict or ndarray
        Acceptance probabilities for the Metropolis algorithm.
    n : int
        Linear size of the spin lattice (length x length x lenght).

    Returns
    -------
    S : float
        Average system's order parameter.
    """
    N = n*n*n
    
    for _ in range(equilibrium_time): 
        metropolis_sweep(states, tensors, probs, n) 

    S = 0.0 #Sets m_squared 

    for _ in range(measurement_time): 
        metropolis_sweep(states, tensors, probs, n) 
        
        unique, counts = np.unique(states, return_counts=True)
        S += max(counts) / N

    S /= measurement_time 
    S = (3/2)*S - (1/2)

    return S

class MaierSaupe_Monte_Carlo: 
    def __init__(
        self, 
        length: int = None, 
        t_min: float = None, 
        t_max: float = None, 
        t_step: float = None) -> None:
        super().__init__()  

        self.length = length
        self.t_min = t_min 
        self.t_max = t_max 
        self.t_step = t_step 

        # Define the 3 possible states
        self.tensors = np.array([
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
        
        self.main() 

    def main(self): 
        #Initialize vectors to export data 
        temperature = [] 
        Sv_avg = [] 
        Sv_error = []

        np.random.seed(int(time.time())) 

        #Initialize Variabels 
        N = self.length*self.length*self.length #Lattice size
        equilibrium_time = 5*N #Equilibrium time 
        measurement_time = 30*N #Measurement time
        number_runs = 50 #Number of experiments

        number_t = int(round((self.t_max - self.t_min) / self.t_step)) + 1 
        
        for t in range(number_t): 
            temp = self.t_max - self.t_step*t #Starts at the higher T 
            beta = float(1/temp) 
            probs=set_tabs(beta) #Pre setting transition probabilities
            
            # Initialize with random configuration
            states = np.random.choice([0,1,2], size=self.length*self.length*self.length)   
            
            # Define S average and error 
            S_avg = 0.0 
            S_error = 0.0  

            #Call for concurrency
            with ProcessPoolExecutor() as executor: 
                #Submit all runs                
                futures = [executor.submit(parallel_run, equilibrium_time, measurement_time, 
                                   np.copy(states), self.tensors, probs, self.length) for _ in range(number_runs)]
                results = [f.result() for f in futures] 
                
                #Averaging the results
                S_avg = np.mean(results)
                S_error = np.sqrt(np.var(results, ddof=1) / number_runs)

            #Show results on terminal
            print(f'-----------\nTemperature:{temp:9.5f}') 
            print(f'Length:{self.length:5d}')
            print(f'S_avg:{S_avg:9.5f}')
            print(f'S_err: {S_error:9.5f}\n-----------') 
            
            #Append results to save
            temperature.append(temp) 
            Sv_avg.append(S_avg)
            Sv_error.append(S_error) 

        #Save results
        np.savetxt(f'Data/MaierSaupe_MonteCarlo_L{self.length}.csv',  [p for p in zip(temperature, Sv_avg, Sv_error)], delimiter=',', fmt='%6.5f',
            header='{0:^5s},{1:^7s},{2:^9s}'.format('Temperature','S average','S error'),comments='')
 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Monte Carlo simulation of the 3-dimensional Maier Saupe Model')

    #Positional Arguments
    parser.add_argument('length', type=int, help='Lattice length used to define lattice size (length*length*length)')
    parser.add_argument('t_min', type=float, help='Temperature lower bound')
    parser.add_argument('t_max', type=float, help='Temperature higher bound')
    parser.add_argument('t_step', type=float, help='Simulation temperature step')
    args = parser.parse_args()

    length = args.length
    t_min = args.t_min 
    t_max = args.t_max 
    t_step = args.t_step  
    
    MaierSaupe_Monte_Carlo(length, t_min, t_max, t_step)

