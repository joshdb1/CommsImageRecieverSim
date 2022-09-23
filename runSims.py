import numpy as np
from rx import rx
import math
from matplotlib import pyplot as plt

## setup (taken from com_sim_for_students_2022.py)
# Set parameters
f_eps        = 0.0 # Carrier frequency offset percentage (0 = no offset)
Phase_Offset = 0.0 # Carrier phase offset (0 = no offset)
t_eps        = 0.0 # Clock freqency offset percentage (0 = no offset)
T_offset     = 0.0 # Clock phase offset (0 = no offset)
Ts = 1             # Symbol period
N = 4              # Samples per symbol period

# Select modulation type
# Use 8 bits per symbol and 256 square QAM
B = 8            # Bits per symbol (B should be even: 8, 6, 4, 2)
bits2index = 2**np.arange(B-1,-1,-1)
M = 2 ** B       # Number of symbols in the constellation
Mroot = math.floor(2**(B/2))
a = np.reshape(np.arange(-Mroot+1,Mroot,2),(2*B,1))
b = np.ones((Mroot,1))
LUT = np.hstack((np.kron(a,b), np.kron(b,a)))

# Scale the constellation to have unit energy
Enorm = np.sum(LUT ** 2) / M
LUT = LUT/math.sqrt(Enorm)

# Assuming a known unique word
uw = np.array([162,29,92,47,16,112,63,234,50,7,15,211,109,124,
                239,255,243,134,119,40,134,158,182,0,101,
                62,176,152,228,36])
uw_len = uw.size
uw = uw.reshape(uw_len,)

uwsym = LUT[uw,:]

Eave = 1
Eb = Eave/B
EbN0dB = 30 # SNR in dB
N0 = Eb*10**(-EbN0dB/10)
nstd = math.sqrt(N0/2) # Noise standard deviation

# Build the list of four possible UW rotations
angles = 2*math.pi*np.arange(0,4)/4
uwrotsyms = np.zeros((uw_len,2,4))

# Generate received signal with a clock frequency offset
print('Generating SRRC ... ')
EB = 0.7  # Excess bandwidth
To = (1+t_eps)
srrc = []
if (t_eps == 0):
    Lp = 12
    t = np.arange(-Lp*N,Lp*N+1) /N + 1e-8  # +1e-8 to avoid divide by zero
    tt = t + T_offset
    srrc = ((np.sin(math.pi*(1-EB)*tt)+ 4*EB*tt * np.cos(math.pi*(1+EB)*tt))
        /((math.pi*tt)*(1-(4*EB*tt)**2)))
    srrc = srrc/math.sqrt(N)
    
Omega0 = math.pi/2*(1+f_eps)
    
print('setup done.\n')

####################################################
# TEST
####################################################
file = "test_2022"
s = np.genfromtxt(file)

print("Running " + file)
img, phErrorSig, phEst, timeLfOut, mu, decX, decY = rx(s, srrc, N, LUT, uw, Omega0, Ts)

plt.figure()
plt.imshow(img, 'gray')
plt.title(file)
plt.savefig("figs/" + file + ".png")

plt.figure()
plt.plot(decX, decY, '.')
plt.title(file + " decision vars in signal space")
plt.savefig("figs/" + file + "_decisions.png")

####################################################
# SIM1
####################################################

file = "sim1_2022"
s = np.genfromtxt(file)

print("Running " + file)
img, phErrorSig, phEst, timeLfOut, mu, decX, decY = rx(s, srrc, N, LUT, uw, Omega0, Ts)

plt.figure()
plt.imshow(img, 'gray')
plt.title(file)
plt.savefig("figs/" + file + ".png")

plt.figure()
plt.plot(decX, decY, '.')
plt.title(file + " decision vars in signal space")
plt.savefig("figs/" + file + "_decisions.png")

####################################################
# SIM2
####################################################

file = "sim2_2022"
s = np.genfromtxt(file)

print("Running " + file)
img, phErrorSig, phEst, timeLfOut, mu, decX, decY = rx(s, srrc, N, LUT, uw, Omega0, Ts)

plt.figure()
plt.imshow(img, 'gray')
plt.title(file)
plt.savefig("figs/" + file + ".png")

plt.figure()
plt.plot(decX, decY, '.')
plt.title(file + " decision vars in signal space")
plt.savefig("figs/" + file + "_decisions.png")

plt.figure()
plt.plot(phEst)
plt.title(file + " phase estimate")
plt.savefig("figs/" + file + "_phEst.png")

plt.figure()
plt.plot(phErrorSig)
plt.title(file + " phase estimate")
plt.savefig("figs/" + file + "_phError.png")

####################################################
# SIM3
####################################################

file = "sim3_2022"

s = np.genfromtxt(file)

print("Running " + file)
img, phErrorSig, phEst, timeLfOut, mu, decX, decY = rx(s, srrc, N, LUT, uw, Omega0, Ts)

plt.figure()    
plt.imshow(img, 'gray')
plt.title(file)
plt.savefig("figs/" + file + ".png")

plt.figure()
plt.plot(decX, decY, '.')
plt.title(file + " decision vars in signal space")
plt.savefig("figs/" + file + "_decisions.png")

plt.figure()
plt.plot(phEst)
plt.title(file + " phase estimate")
plt.savefig("figs/" + file + "_phEst.png")

plt.figure()
plt.plot(phErrorSig)
plt.title(file + " phase estimate")
plt.savefig("figs/" + file + "_phError.png")

####################################################
# SIM4
####################################################

file = "sim4_2022"

s = np.genfromtxt(file)

print("Running " + file)
img, phErrorSig, phEst, timeLfOut, mu, decX, decY = rx(s, srrc, N, LUT, uw, Omega0, Ts)

plt.figure()    
plt.imshow(img, 'gray')
plt.title(file)
plt.savefig("figs/" + file + ".png")

plt.figure()
plt.plot(decX, decY, '.')
plt.title(file + " decision vars in signal space")
plt.savefig("figs/" + file + "_decisions.png")

plt.figure()
plt.plot(timeLfOut, '.')
plt.title(file + " filtered timing error")
plt.savefig("figs/" + file + "_timeError.png")

plt.figure()
plt.plot(mu, '.')
plt.title(file + " mu (fractional import)")
plt.savefig("figs/" + file + "_mu.png")

####################################################
# SIM5
####################################################

file = "sim5_2022"

s = np.genfromtxt(file)

print("Running " + file)
img, phErrorSig, phEst, timeLfOut, mu, decX, decY = rx(s, srrc, N, LUT, uw, Omega0, Ts)

plt.figure()    
plt.imshow(img, 'gray')
plt.title(file)
plt.savefig("figs/" + file + ".png")

plt.figure()
plt.plot(decX, decY, '.')
plt.title(file + " decision vars in signal space")
plt.savefig("figs/" + file + "_decisions.png")

plt.figure()
plt.plot(timeLfOut, '.')
plt.title(file + " filtered timing error")
plt.savefig("figs/" + file + "_timeError.png")

plt.figure()
plt.plot(mu, '.')
plt.title(file + " mu (fractional import)")
plt.savefig("figs/" + file + "_mu.png")

####################################################
# SIM6
####################################################

file = "sim6_2022"

s = np.genfromtxt(file)

print("Running " + file)
img, phErrorSig, phEst, timeLfOut, mu, decX, decY = rx(s, srrc, N, LUT, uw, Omega0, Ts)

plt.figure()    
plt.imshow(img, 'gray')
plt.title(file)
plt.savefig("figs/" + file + ".png")

plt.figure()
plt.plot(decX, decY, '.')
plt.title(file + " decision vars in signal space")
plt.savefig("figs/" + file + "_decisions.png")

plt.figure()
plt.plot(phEst)
plt.title(file + " phase estimate")
plt.savefig("figs/" + file + "_phEst.png")

plt.figure()
plt.plot(phErrorSig)
plt.title(file + " phase estimate")
plt.savefig("figs/" + file + "_phError.png")

plt.figure()
plt.plot(timeLfOut, '.')
plt.title(file + " filtered timing error")
plt.savefig("figs/" + file + "_timeError.png")

plt.figure()
plt.plot(mu, '.')
plt.title(file + " mu (fractional import)")
plt.savefig("figs/" + file + "_mu.png")
