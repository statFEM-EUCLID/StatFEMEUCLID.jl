[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://statfem-euclid.github.io/StatFEMEUCLID.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://statfem-euclid.github.io/StatFEMEUCLID.jl/dev/)

# StatFEMEUCLID

This package provides tooling for statFEM-EUCLID[^1], which combines the statistical finite element method (statFEM)[^2] and EUCLID (Efficient Unsupervides Constitutive Law Identification and Discovery)[^3] for unsupervised constitutive model discovery.
Instead of 'hard-wiring' a single finite element software, the package is able to use any solver as a black box
through a [UM-Bridge](https://um-bridge-benchmarks.readthedocs.io/en/docs/) interface.

## Features
- A high(er)-level interface to UM-Bridge for sampling an FEM black box
- Construction of PCE surrogates from samples
- Data assimilation based on the statFEM method, see Example201
- Parameter identification based on the virtual fields method or EUCLID (WIP: in refactoring)



## License
StatFEMEUCLID.jl is licensed under the MIT License. 


## References
[^1]: V. Narouie, JH. Urrea-Quintero, F. Cirak, H. Wessels *Unsupervised Constitutive Model Discovery from Sparse and Noisy Data*  https://doi.org/10.1016/j.cma.2025.118722
[^2]: M. Girolami, E. Febrianto, G. Yin, F. Cirak *The statistical finite element method (statFEM) for coherent synthesis of observation data and model predictions*  https://doi.org/10.1016/j.cma.2020.113533
[^3]: M. Flaschel, S. Kumar, L. De Lorenzis  *Unsupervised discovery of interpretable hyperelastic constitutive laws* https://doi.org/10.1016/j.cma.2021.113852
