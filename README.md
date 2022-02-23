# Cycles-EOW-simulations

This repository contains the python script to run EOW simulations using Cycles, for different crops and climate scenarios.
A look-up table provides corresponding soil files and weather files for all locations in the world where a certain crop is planted.
The python script will analyze the weather files to find the potential planting window based on air temperature.
Then the crop is planted for each month in the planting window, to identify the optimum month for planting.
The results are compiled into a summary file in the `summary` directory.

To run the simulations, soil file and weather file archives need to be provided in the `data` directory.

To run the code,

```shell
module load anaconda3

./run-cycles.py --crop Maize --scenario nw_cntrl_03
```
