# Brownian_Dynamics_CUDA

Brownian Dynamics simulation for mesoscale chromatin fiber with CUDA implementation.

## Data

Test data are provided in `test_data`, modify `setup.txt` to specify inputs.

## Simulation

Compile the code with ``./compile.sh`` under ``BD_cuda`` directory.

Put the binary file ``code`` under ``test_data/`` directory, and run ``./code`` to perform the BD simulation.  

## Output

The simulated structure is recorded in ``out.xyz`` in ``XYZ`` file format.

## Reference

Z. Li, S. Portillo-Ledesma, and T. Schlick. (2022) Brownian Dynamics Simulations of Mesoscale Chromatin Fibers. Biophys. J., 122, 1–14.
