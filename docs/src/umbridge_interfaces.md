# UMBridge interfaces

## FEM forward model
The forward model should solve the deterministic 
boundary value problem (DBVP) for a given traction $T_N$ 
and (optionally) a set of material parameters $\kappa\in\mathbb{R}^k$
and return the solution $u_h\in\mathbb{R}^{n_{dof}*d}$ at all degrees of freedom.

For use within this package the model should have the following parameters
- inputSizes: 
    - `[1]` for simple cases with internally fixed $\kappa$
    - `[1,k]` for cases with varying parameters (can be config dependent)
- outputSizes:
    - `[n_dofs]` for 1D
    - `[n_dofs,n_dofs]` for 2D
    - `[n_dofs,n_dofs,n_dofs]` for 3D
- evaluate should be a function that returns the DBVP solution for the given parameters.

## Projection model
The projection model is used to assemble the projection matrix
$H$ which maps from nodal points to sensor locations.
The nonzero entries of a row of $H$ are the 
nonzero basis functions of the respective FEM discretization.
To keep the interface simple, the projection model 
receives a single sensor point $X_s$ and returns 
a vector containing the DoF ids $i$ and corresponding basis values $\phi_i(X_s)$.
As an example, on a triangular 2D mesh with bilinear P1 elements,
the input size is`[2]` (x,y) and ansatz for any point within the domain
has at most 3 nonzero basis values, such that the output sizes should be `[3,3]`.

Since it is not necessarily trivial to get this information from a
specific FEM solver, we also provide an extension based on ExtendableFEM.jl that can calculate $H$ based on a given mesh.

## Residual model (to be written)
