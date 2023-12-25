# A Mean-Field Treatment of Vacuum Fluctuations in Strong Light–Matter Coupling
Authors: Ming-Hsiu Hsieh, Alex Krotz, Roel Tempelaar

*J. Phys. Chem. Lett.* **2023**, 14, 5, 1253–1258

Arxiv: https://arxiv.org/abs/2211.15949

Publication: https://pubs.acs.org/doi/full/10.1021/acs.jpclett.2c03724

## Usage of the code
The code has been run under Python 3.8.

Required packages: Numpy, Numba, Ray, time

To run a simulation with MF or DC-MF:
```
python3 main.py input.txt
```
In line 29 of main.py, one can run the MF dynamics by calling `runCalc` in `mixQC_2atoms` and DC-MF dynamics by calling `runCalc` in `mixQC_2atoms_MFDC`.

To get the results and plots for MF dynamics:
```
python3 analyze_Hoff.py
```
To get the results and plots for DC-MF dynamics:
```
python3 analyze_MFDC.py
```

Alternatively, to run a simulation with CISD (note that this also reads input.txt):
```
python3 exact_CISD_Natom_RK4.py
```
To calculate the field intensity within CISD:
```
python3 calc_ele_int_elewise.py
```

## Explanation to each file
+ input.txt: input file where one sets all parameters
+ main.py: the main executive file
+ mixQC_2atoms.py: functions for MF dynamics
+ mixQC_2atoms_MFDC.py: functions for DC-MF dynamics
+ analyze_MFDC.py: analyze the DC-MF results and get figures
+ analyze_Hoff.py: analyze the MF results and get figures
+ exact_CISD_Natom_RK4.py: executive file for the CISD result
+ calc_ele_int_elewise.py: calculating the electric field intensity from CISD.

