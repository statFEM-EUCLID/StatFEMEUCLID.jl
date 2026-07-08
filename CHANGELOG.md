# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.2.0] - 2026-07-08

Breaking change: `covariance` functions in Sampling and PCE renamed to `cov` to match respective functions in `Statistics` std library.

### Added
- Sampling and forward model evaluation now allow for extra parameters to be send (e.g. material parameters)
- SensorSets and StatFEM submodules 
- Example201 explaining use of StatFEM
- Public submodule functions are now reexported by main module
- Test for Example101 added with Mocking.jl
- Aqua.jl quality checks

### Changed
- `mean`, `var` and `cov` now correctly extend respective methods from `Statistics.jl`


## [0.1.0] - 2026-03-30

Initial release providing a first example
