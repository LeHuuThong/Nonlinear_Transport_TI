# Nonlinear Transport in Topological Insulators

## Overview

This repository contains the computational code and data analysis tools for studying nonlinear Hall transport in topological insulators. The code implements tight-binding models and quantum transport calculations to investigate Hall conductances and related phenomena in quantum anomalous Hall (QAH) systems and axion insulators.

## Description

The codebase provides:

- **Hamiltonian Construction**: Implementation of tight-binding models for topological insulators, including two-surface models and quantum anomalous Hall systems
- **Transport Calculations**: First and second-order Hall conductance calculations using the Kwant quantum transport library
- **Data Analysis**: Tools for analyzing transport properties, density of states, and band structures
- **Visualization**: Plotting utilities for generating publication-quality figures

## Key Components

### Core Modules

- `HallTransport.py`: Main transport calculation classes including `HallResistance_1st` and `HallResistance_2nd`
- `Hamiltonians.py`: Implementation of various Hamiltonian models for topological systems
- `Analysis.py`: High-level analysis functions for systematic studies
- `common.py` / `common1.py`: Utility functions and shared computational routines

### Data and Results

- `Data/`: Pickle files containing computed transport data
- `Figures/`: Generated publication figures
- `Note_Figures.ipynb`: Jupyter notebook for figure generation and analysis

## Dependencies

The code requires the following Python packages (see `requirements.txt` for specific versions):

- `kwant`: Quantum transport calculations
- `numpy`: Numerical computations
- `scipy`: Scientific computing utilities
- `matplotlib`: Plotting and visualization
- `dask`: Parallel computing support
- `jupyter`: For running notebooks

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd NonlinearTransportTI
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Transport Calculation

```python
import HallTransport as HT
import Hamiltonians

# Create Hamiltonian
hamiltonian = HT.HamiltonianBuilder(Hoppings, W=20, L=20, nW=2, nL=5, 
                                   eU_values=np.zeros((2,5)))
hamiltonian.create_system()

# Calculate first-order Hall conductance
hall1 = HT.HallResistance_1st(hamiltonian, energy=-0.1)
hall1.evaluate_smatrix()
hall1.evaluate_Hall_resistance()

# Calculate second-order nonlinear response
hall2 = HT.HallResistance_2nd(hamiltonian, energy=-0.1)
hall2.evaluate_all()
```

### Analysis and Visualization

See `Note_Figures.ipynb` for detailed examples of data analysis and figure generation.

## Citation

If you use this code in your research, please cite the associated publication. See [CITATION.cff](CITATION.cff) for machine-readable citation information, or use the following BibTeX:

```bibtex
@article{YourPaper2025,
  title={[Your Paper Title]},
  author={Huu-Thong Le and Chao-Xing Liu},
  journal={[Journal Name]},
  volume={[Volume]},
  pages={[Pages]},
  year={2025},
  doi={[DOI]}
}
```

## License and Copyright

Copyright (c) 2024 Huu-Thong Le, Chao-Xing Liu

This software is released under the MIT License. See the [LICENSE](LICENSE) file for full license details.

## Academic Use and Attribution

This code was developed as part of academic research. When using this software:

1. **Attribution Required**: Please cite the associated publication(s) listed above
2. **Academic Courtesy**: Notify the authors if you use this code in your research
3. **Derivative Works**: If you modify or extend this code significantly, please acknowledge the original source
4. **Data Policy**: The computed data in the `Data/` directory is provided for reproducibility. Please cite appropriately if used directly.

## Contact Information

For questions about the code or collaboration opportunities, please contact:

- **Primary Author**: [Prof. Chao-Xing Liu] - [cxl56@psu.edu]
- **Institution**: [Department of Physics - Pennsylvania State University]

## Acknowledgments

We acknowledge the use of the Kwant quantum transport package and thank the developers for making it freely available.

## Version History

- **v1.0**: Initial release accompanying publication

## Technical Notes

### System Requirements
- Python 3.7+
- Memory: Minimum 8GB RAM for typical calculations
- Storage: ~100MB for code and sample data

### Performance Considerations
- Large system calculations may require significant computational resources
- Parallel computing support via Dask for improved performance
- Consider using HPC resources for extensive parameter sweeps

### Known Issues
- For bug reports, please open an issue in the repository

---

**Disclaimer**: This software is provided for research purposes. While we strive for accuracy, users should validate results for their specific applications.