#!/usr/bin/env python3   

"""
Monte Carlo simulation of the 2-dimensional Ising model 

"""
import numpy as np
import time
import argparse 


def metropolis_sweep(spins, length, probs): 
    """
    Performs one Monte Carlo sweep using the Metropolis algorithm

    """ 
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
    """
    Precomputes the Metropolis transition probabilities

    """
    prob = np.zeros(17)
    for i in range(-8, 9):
        prob[i + 8] = np.exp(-beta*i) if i > 0 else 1  
    
    return prob

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

        N = self.length*self.length #Lattice size
        equilibrium_time = 5*N
        measurement_time = 30*N 
        number_runs = 50 

        number_t = int(round((self.t_max - self.t_min) / self.t_step)) + 1 

        for t in range(number_t): 
            temp = self.t_max - self.t_step*t #Starts at the higher T 
            beta = float(1/temp) 
            probs=set_tabs(beta) 
            
            # Initialize all spins up
            spins = np.ones((self.length, self.length), dtype=int) 

            # Define m_squared's average and error 
            m2_avg = 0.0 
            m2_err = 0.0  

            for _ in range(number_runs): 
                for _ in range(equilibrium_time): 
                    metropolis_sweep(spins, self.length, probs) 

                m2=0.0 #Sets m_squared 

                for _ in range(measurement_time): 
                    metropolis_sweep(spins, self.length, probs) 
                    m = np.sum(spins) 
                    m2 += m * m 

                m2 /= measurement_time * N * N
                m2_avg += m2
                m2_err += m2 * m2

            m2_avg /= number_runs
            m2_err /= number_runs
            m2_err = np.sqrt(abs(m2_err - m2_avg**2) / (number_runs - 1))

            #Show results on terminal
            print(f'-----------\nTemperature:{temp:9.5f}') 
            print(f'Length:{self.length:5d}')
            print(f'm2_avg:{m2_avg:9.5f}')
            print(f'm2_err: {m2_err:9.5f}\n-----------') 
            
            #Append values to save
            temperature.append(temp) 
            m_squared_avg.append(m2_avg)
            m_squared_error.append(m2_err) 

        #Save results
        np.savetxt(f'Data/Ising_MonteCarlo_L{self.length}.csv',  [p for p in zip(temperature, m_squared_avg, m_squared_error)], delimiter=',', fmt='%6.5f',
            header='{0:^5s},{1:^7s},{2:^9s}'.format('Temperature','m2 average','m2 error'),comments='')
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Monte Carlo simulation of the 2-dimensional Ising model')

    #Positional Arguments
    parser.add_argument('length', type=int, help='Lattice length used to define lattice size (length*length). Must be lower than 32')
    parser.add_argument('t_min', type=float, help='Temperature lower bound')
    parser.add_argument('t_max', type=float, help='Temperature higher bound')
    parser.add_argument('t_step', type=float, help='Simulation temperature step')
    args = parser.parse_args()

    length = args.length
    t_min = args.t_min 
    t_max = args.t_max 
    t_step = args.t_step  
    
    Ising_Monte_Carlo(length, t_min, t_max, t_step)

