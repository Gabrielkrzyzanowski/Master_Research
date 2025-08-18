#!/usr/bin/env python3   
"""
Monte Carlo simulation of the 2D Ising model.

This script simulates a square lattice of spins using the Metropolis algorithm.
It computes the average squared magnetization at different temperatures and
saves the results to a CSV file. Parallel execution is used to speed up 
multiple independent runs.

Usage
-----
Run the script from the command line with the following arguments:

    python script_name.py length t_min t_max t_step

where:
    length : int    -> Lattice linear size (number of spins per side)
    t_min  : float  -> Minimum temperature
    t_max  : float  -> Maximum temperature
    t_step : float  -> Temperature step

The output CSV file will contain columns:
    Temperature, m2 average, m2 error
"""
import numpy as np # type: ignore
import time
import argparse 
from concurrent.futures import ProcessPoolExecutor 

def metropolis_sweep(spins, probs, n):
    """
    Performs one Monte Carlo sweep using the Metropolis algorithm
    """ 
    for i in range(len(spins)-1): 
        de = 2 * spins[i] * ( 
            spins[((i // n + 1) % n) * n + (i % n)] + 
            spins[((i // n - 1) % n) * n + (i % n)] + 
            spins[(i // n) * n + ((i % n + 1) % n)] + 
            spins[(i // n) * n + ((i % n - 1) % n)] 
        )
        if de <=0: 
            spins[i] *= -1
        else:
            if np.random.random() < probs[de + 8]:
                spins[i] *= -1

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
    prob = np.zeros(17)
    for i in range(-8, 9):
        prob[i + 8] = np.exp(-beta*i) if i > 0 else 1  

    return prob

def parallel_run(equilibrium_time, measurement_time, spins, probs, length):
    """
    Run a Monte Carlo simulation to compute the average squared magnetization.

    Parameters
    ----------
    equilibrium_time : int
        Number of sweeps to equilibrate the system.
    measurement_time : int
        Number of sweeps over which measurements are taken.
    spins : ndarray
        2D array representing the spin configuration.
    probs : dict or ndarray
        Acceptance probabilities for the Metropolis algorithm.
    length : int
        Linear size of the spin lattice (length x length).

    Returns
    -------
    m2 : float
        Average squared magnetization per site.
    """
    N = length*length
    
    for _ in range(equilibrium_time): 
        metropolis_sweep(spins[0], probs, length) 

    m2=0.0 #Sets m_squared 

    for _ in range(measurement_time): 
        metropolis_sweep(spins[0], probs, length) 
        m = np.sum(spins) 
        m2 += m * m 

    m2 /= measurement_time * N * N
    return m2

class Ising_Monte_Carlo: 
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

        self.main() 

    def main(self): 
        #Initialize vectors to export data 
        temperature = [] 
        m_squared_avg = [] 
        m_squared_error = []

        np.random.seed(int(time.time())) 

        #Initialize Variabels 
        N = self.length*self.length #Lattice size
        equilibrium_time = 5*N #Equilibrium time 
        measurement_time = 30*N #Measurement time
        number_runs = 50 #Number of experiments

        number_t = int(round((self.t_max - self.t_min) / self.t_step)) + 1 

        for t in range(number_t): 
            temp = self.t_max - self.t_step*t #Starts at the higher T 
            beta = float(1/temp) 
            probs=set_tabs(beta) #Pre setting transition probabilities
            
            # Initialize all spins up
            spins = np.ones((1, N), dtype=int) 

            # Initialize all spins randomly 
            #spins = np.random.choice([1, -1], size=(1, N))
            
            # Define m_squared's average and error 
            m2_avg = 0.0 
            m2_err = 0.0  

            #Call for concurrency
            with ProcessPoolExecutor() as executor: 
                #Submit all runs                
                futures = [executor.submit(parallel_run, equilibrium_time, measurement_time,
                                   np.copy(spins), probs, self.length) for _ in range(number_runs)]
                results = [f.result() for f in futures] 
                
                #Averaging the results
                m2_avg = np.mean(results)
                m2_err = np.sqrt(np.var(results, ddof=1) / number_runs)

            #Show results on terminal
            print(f'-----------\nTemperature:{temp:9.5f}') 
            print(f'Length:{self.length:5d}')
            print(f'm2_avg:{m2_avg:9.5f}')
            print(f'm2_err: {m2_err:9.5f}\n-----------') 
            
            #Append results to save
            temperature.append(temp) 
            m_squared_avg.append(m2_avg)
            m_squared_error.append(m2_err) 

        #Save results
        np.savetxt(f'Data/Ising_MonteCarlo_L{self.length}.csv',  [p for p in zip(temperature, m_squared_avg, m_squared_error)], delimiter=',', fmt='%6.5f',
            header='{0:^5s},{1:^7s},{2:^9s}'.format('Temperature','m2 average','m2 error'),comments='')
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Monte Carlo simulation of the 2-dimensional Ising model')

    #Positional Arguments
    parser.add_argument('length', type=int, help='Lattice length used to define lattice size (length*length)')
    parser.add_argument('t_min', type=float, help='Temperature lower bound')
    parser.add_argument('t_max', type=float, help='Temperature higher bound')
    parser.add_argument('t_step', type=float, help='Simulation temperature step')
    args = parser.parse_args()

    length = args.length
    t_min = args.t_min 
    t_max = args.t_max 
    t_step = args.t_step  
    
    Ising_Monte_Carlo(length, t_min, t_max, t_step)

