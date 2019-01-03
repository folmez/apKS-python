# apKS-python
Python scripts for the apKS project

TO-DO

- Implement plotting theoretical PDF plotting in testing for EPL samples for apKS testing
- Implement data type (REAL or INTS) detection in apKS
- Go over discrete data (INTS) handling, src and tests
- Implement the experiments in the EPL paper
- Make a main method that parses command line input, make it read a .dat or .txt file
- Describe how to set up the python code in the command line, what packages are needed, how to develop setup.py

- January 3, 2019: Power-law fit drawing implemented into papod.py, tests updated accordingly
- December 29, 2018: apKS completely implemented along with tests
- December 28, 2018: Penalized KS method for power-law fitting is completely implemented
- December 27, 2018: EPL3 fully implemented.
- December 26, 2018: EPL2 implemented. Input interval is tested for errors in synthetic data generation.
- December 25, 2018: p-value estimation is implemented
- December 25, 2018: KS method for bounded power-law fitting implemented
- December 24, 2018: Semiparametric bootstrap sample generation implemented
- December 24, 2018: Repository organized
- November 10, 2018: Power-law exponent estimation using maximum-likelihood is added (estexp). Choosing an almost logarithmically equally spaced subset of a sample is added (elspd). Estimating KS distance between empirical sample CDF and theoretical power-law PDF is added (estKS). Also, a testing module (test_apKS.py) is added that works in the pytest framework.
- November 6, 2018: Theoretical PDF drawing of random samples is added.
- November 5, 2018: Power-law fit PDF drawing on top of approximate PDF is added.
- November 4, 2018: Approximate PDF drawing is added.
- November 1, 2018: Exact power-law 1 (EPL1) random sample generation is added.

- The apKS project was originally written in MatLab. apKS-python is the same project written in Python. A side goal of working on this github repository is for me become more familiar with interacting with github from terminal and Python.
