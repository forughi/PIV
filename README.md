# Particle Image Velocimetry (PIV) Code
A flexible and hackable Particle Image Velocimetry (PIV) code in Python and MATLAB.


# Notes:
- The Python version of the code (Python_Code.py) has been updated frequently. The MATLAB version is much older. Please use the Python code if you can.
- In some cases, Numba makes the code run much faster (up to 2.2 times). If you don't want to use Numba, comment out lines 12 and 15.
- If you want to call the PIV code as a function in your code, please see the contents of the "Python_Function" folder in the current repository. There is a sample code ("sample.py") that can show you how to call the function from "piv_lib.py".
- Joblib has been used for parallel processing. It can significantly improve performance and speed up the processing. The "cores" variable defines the number of parallel jobs, and the default is set to the maximum number of available cores. To limit the number of cores or disable parallel processing, adjust it as shown in the code. 
- You can select to use FFT or legacy cross-correlation by setting the variable use_fft to True or False, respectively. The FFT method is much faster in most cases. 
