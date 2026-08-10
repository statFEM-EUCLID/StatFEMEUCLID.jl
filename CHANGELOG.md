# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Fixed error in generation of synthetic data in Example201 
- Fixed `draw_FEM_samples` for `solution_index=:`

### [0.2.3] - 2026-08-08

### Added
- Result plots for Example201
- Compat for Polychaos 2
- Functionality for EqualSensorSet, `read`, `project` and `apply_noise!`

### Changed
- Reduced number of sensors in Example201
- Expanded description of Example201
- setup-python action version to 7

## [0.2.2] - 2026-07-10

### Added 
- ExtendableFEMBase.jl-based extension to provide a client-side calculation of the projection matrix

## [0.2.1] - 2026-07-08

Small bugfixes in Example201

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
