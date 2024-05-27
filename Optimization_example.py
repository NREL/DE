import DE
from numba import jit
import numpy as np
import time

@jit(nopython=True)
def ackley(indv):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (indv[0]**2 + indv[1]**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * indv[0]) + np.cos(2 * np.pi * indv[1])))
    value = term1 + term2 + np.e + 20
    return value

if __name__ == '__main__':

    lbound = np.array([-10,-10])
    ubound = np.array([10,10])

    total_time = 0
    timed_runs = 10
    parallel_runs = 10
    for i in range(timed_runs):
        start_time = time.time()
        score_list, solution_list = DE.parallel(ackley,
                                                runs=parallel_runs, popsize=100, gmax=10000, 
                                                lower_bound=lbound, upper_bound=ubound)
        end_time = time.time()
        total_time += end_time - start_time
        
        with open(f"./Ackley/Ackley_{i}.txt", "w") as text_file:
                text_file.write("Run Score X Y\n" )
                for j in range(parallel_runs):
                    text_file.write((
                                    f"{j:d} "
                                    f"{score_list[j,0]:.3e} "
                                    f"{solution_list[j,0]:.3e} "
                                    f"{solution_list[j,1]:.3e}\n"
                        ))
    average_time = total_time / timed_runs
    print(f"Average execution time over {timed_runs} runs: {average_time} seconds")