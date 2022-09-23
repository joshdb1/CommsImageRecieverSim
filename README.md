# CommsSystemRecieverSim

This program simulates the image communications system receiver shown below:

![alt text](https://github.com/joshdb1/CommsSystemRecieverSim/blob/master/figs/sysDiagram.PNG?raw=true)

The receiver implementation is in `rx.py`. The simulation includes demodulation, matched filtering, and phase and timing synchronization using PLL's. The receiver takes a raw 1D signal as input and outputs the received 2D image.

To run the simulations, run the command (using Python 3.9 or later): `python ./runSims.py`
