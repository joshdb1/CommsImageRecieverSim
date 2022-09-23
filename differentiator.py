import numpy as np

def differentiator(x, L, T, A=1):
    L = np.float32(L)
    filtLen = np.int32(2 * L + 1)
    nlst = np.arange(-L, L + 1, 1)
    h = np.zeros(filtLen)
    w = np.blackman(filtLen)
    
    for i in range(0, filtLen):
        n = nlst[i]
        if n != 0:
            h[i] = (1 / T) * (-1)**n / n
        else:
            h[i] = 0
       
    # generate the FIR filter
    hFir = h * w
    
    xp = np.convolve(x, hFir)

    return xp, hFir

def delayer(x, L):
    delay = np.zeros(2 * L + 1)
    delay[L] = 1
    xdel = np.convolve(x, delay)
    return xdel, delay

                