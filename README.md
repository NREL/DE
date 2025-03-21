Copyright (c) 2024 Alliance for Sustainable Energy, LLC.

NREL software record: SWR-24-59

## DE (Differential Evolution)
Numba-compatible self-adaptive differential evolution algorithm for optimization tasks. This implementation is based on the method introduced by Brest, Janez, et al., in their study "Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems," published in IEEE Transactions on Evolutionary Computation, Vol. 10, No. 6, 2006, pp. 646-657.

### SDM_Extraction
**Script**: Facilitates the extraction of single diode parameters from IV data, streamlining the analysis process.

### Optimization_example
**Script**: Exemplifies the application of the DE algorithm to minimize the 2D Ackley function

### ackley_results
**Script**: Results from minimizing the Ackley function

### Results_analysis
**Jupyter Notebook**: Employed for generating figures based on the data collected, aiding in the visualization and interpretation of results.

### LambertW Approximation
**Method Evaluation**: Evaluates the main branch of the Lambert W function, approximated via the Newton-Raphson method, which is then compared against results from the SciPy implementation.

## 📚 How to Cite

If you use this repository in your research or find it helpful, please cite the following paper:

```bibtex
@article{febba2025_dd,
  author    = "Fébba, Davi and Egbo, Kingsley and Callahan, William A. and Zakutayev, Andriy",
  title     = "From text to test: AI-generated control software for materials science instruments",
  journal   = "Digital Discovery",
  year      = "2025",
  volume    = "4",
  issue     = "1",
  pages     = "35-45",
  publisher = "RSC",
  doi       = "10.1039/D4DD00143E",
  url       = "http://dx.doi.org/10.1039/D4DD00143E"
}

