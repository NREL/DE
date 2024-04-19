#!/usr/bin/env python

# ------ import packages -----------------#
import numpy as np
from numba import jit
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
import DE
import time


# --- Main Loop ------ #

def clean_raw_data(iv_folder, iv_curve, vmin, vmax, output_folder=None, write_to_file=False, output_filename="cleaned_iv_data.txt"):
    
    """Cleans a txt file with voltage and current data."""
    
    data = np.loadtxt(iv_folder + "/" + iv_curve, usecols=(0, 1))
    
    # Sort the data from reverse to forward bias (sort by voltage in ascending order)
    truncated_data = data[(data[:, 0] >= vmin) & (data[:, 0] <= vmax)]
    sorted_data_reverse_to_forward = truncated_data[truncated_data[:, 0].argsort()]
    
    # Separate the re-sorted data into voltage and current arrays
    voltage = sorted_data_reverse_to_forward[:, 0]
    current = sorted_data_reverse_to_forward[:, 1]
    
    if write_to_file and output_folder:
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, output_filename)
        
        # Write the cleaned data to a text file
        np.savetxt(output_file_path, np.column_stack((voltage, current)), fmt='%e', header='Voltage Current')
        print(f"Data written to {output_file_path}")

    return voltage, current

def extract_parameters(iv_folder, results_folder, temperature_celsius, lbound, ubound, vmin, vmax, runs, popsize, gmax):

    os.makedirs(results_folder, exist_ok=True)
    iv_curves = [f for f in listdir(iv_folder) if isfile(join(iv_folder, f))]
    total_files = len(iv_curves)

    for i in range(total_files):  
        voltage, current = clean_raw_data(iv_folder, iv_curves[i], vmin, vmax)
        custom_objective = create_objective_function(voltage, current, temperature_celsius)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Extracting parameters from curve {(i + 1)} of {total_files}: {iv_curves[i]}", flush=True)
        score_list, solution_list = DE.parallel(custom_objective,
                                                runs=runs, popsize=popsize, gmax=gmax, 
                                                lower_bound=lbound, upper_bound=ubound)

        with open(results_folder + "/" + "Results_" + iv_curves[i], "w") as text_file:
            text_file.write("Run Objective I0 n Rs Rsh Offset\n" )
            for j in range(runs):
               text_file.write((
                            f"{j:d} "
                            f"{score_list[j,0]:.3e} "
                            f"{solution_list[j,0]:.3e} "
                            f"{solution_list[j,1]:.3e} "
                            f"{solution_list[j,2]:.3e} "
                            f"{solution_list[j,3]:.3e} "
                            f"{solution_list[j,4]:.3e}\n"
                        ))
               
@jit(nopython=True, fastmath=True)
def lambertw_newton(z, tol=1e-6, max_iter=10):
    w = np.log(z + 1)  # Initial guess
    for i in range(max_iter):
        ew = np.exp(w)
        w_next = w - (w * ew - z) / (ew + w * ew)
        if np.all(np.abs(w_next - w) < tol):
            return w_next
        w = w_next
    return w  # Return the last approximation if convergence criteria are not met

def create_objective_function(voltage, current, temperature_celsius): # Whatever arguments here, as long objective is the returned function, with no arguments.
    """Create a customized objective function with voltage and current data."""
    @jit(nopython=True, fastmath=True)
    def objective(indv):
        charge = 1.602e-19
        boltz = 1.38e-23
        vt = boltz * (temperature_celsius + 273.15) / charge
        adjusted_voltage = voltage + indv[4]
        term1 = 1.0 + indv[2] / indv[3]
        term2 = (indv[0] * indv[2]) / (indv[1] * vt * term1)
        term3 = (indv[2] * indv[0] + adjusted_voltage) / (indv[1] * vt * term1)
        sim = (indv[1] * vt / indv[2]) * lambertw_newton(term2 * np.exp(term3)) - (indv[0] - adjusted_voltage / indv[3]) / term1
        # normalized_error = (np.log(np.abs(current)) - np.log(np.abs(sim))) ** 2
        # return np.sqrt(np.sum(normalized_error) / len(voltage))
        normalized_error = ((current - sim) / (current)) ** 2
        return np.sqrt(np.sum(normalized_error) / len(voltage))
    return objective # Always return objective here, with no arguments.

temperature_list = [500]

for temp in temperature_list:
    iv_folder = "./D2/" + str(temp) + "C/"
    temperature_celsius = temp
    results_folder = iv_folder + "/Timed_execution"
    lbound = np.array([1e-15,   1,   1e-6, 1e-6, -1]) # lower bound for indv[0], indv[1], ...
    ubound = np.array([1,      1e2,   1e3,  1e9,   1])  # upper bound for indv[0], indv[1], ...

    extract_parameters(iv_folder, results_folder, temperature_celsius,
                    lbound, ubound, 
                    vmin=-10, vmax=5, runs=1, popsize=100, gmax=1e4)
     
print("Done!")    