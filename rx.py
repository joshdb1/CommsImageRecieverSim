import numpy as np
from scipy import spatial
from computePllParams import computePllParams
from cubicInterp import cubicInterp
from differentiator import *

import matplotlib.pyplot as plt

def rx(r, pulse, N, LUT, uw, Omega0, Ts):
    uwSz = len(uw)
    bits = 1
    bitsPerSymbol = int(np.log2(len(LUT)))
    imgSyms = []
    
    # Build the list of four possible UW rotations
    angles = 2 * np.pi * np.arange(0, 4) / 4
    uwrotsyms = np.zeros((uwSz, 2, 4))

    uwsym = LUT[uw, :]
    for i in range(angles.size):
        C = np.cos(angles[i])
        S = -np.sin(angles[i])
        G = np.array([[C, -S],[S, C]])
        uwrot = uwsym @ G;  # Rotate the UW symbols
        uwrotsyms[:, :, i] = uwrot; # Save the rotated version

    # demodulate
    I = np.zeros(len(r))
    Q = np.zeros(len(r))
    n = np.arange(I.size)
    I = r * np.sqrt(2) * np.cos(Omega0 * n)
    Q = r * -np.sqrt(2) * np.sin(Omega0 * n)
    
    ## matched filter
    # need to delay the pulse
    pulseDelay, _ = delayer(pulse, 9)
    
    x = np.convolve(I, pulseDelay)
    y = np.convolve(Q, pulseDelay)
    
    ## derivative matched filter
    pulseDer, diffFilt = differentiator(pulse, 9, 1)
    dx = np.convolve(I, pulseDer)
    dy = np.convolve(Q, pulseDer)
    
    constPhaseOffset = 0

    # phase offset rotation matrix
    Cpo = np.cos(constPhaseOffset)
    Spo = -np.sin(constPhaseOffset)
    Gpo = np.array([[Cpo, -Spo],[Spo, Cpo]])
    
    ## phase sync PLL stuff
    # phase sync PLL params
    theta = constPhaseOffset
    k0Phase = 1
    kpPhase = 1.5
    BnPhase = 0.1
    T = Ts
    zetaPhase = 2.5
    k1Phase, k2Phase = computePllParams(zetaPhase, BnPhase, T, k0Phase, kpPhase)
    k2PhaseOut = 0
    
    # phase sync PLL trackers
    phErrorSig = []
    phEstimate = []
    
    ## timing sync PLL stuff
    # timing sync PLL params
    mu = 0
    mu1 = mu
    strobe = 0
    w1 = 2 * mu
    nco = 1
    kpTime = 0.23
    k0Time = -1
    BnTime = 0.003
    zetaTime = 1 / np.sqrt(2)
    k1Time, k2Time = computePllParams(zetaTime, BnTime, T, k0Time, kpTime) 
    v2Time = mu
    eTime = 0
    timeErrorSig = []
    timeLfOut = []
    muk = []
    etak = []
    
    # column data (keeping track of where we are, storing amplitudes/symbols, etc...)
    ncols = 0
    colDataX = []
    colDataY = []
    colDataSyms = []
    colIdx = 0
    k = -1
    
    decX = []
    decY = []
    
    # loop through sampled values
    for j in range(len(x)):
        temp = nco - w1
        if temp < 0:
            strobe = 1
            mu = nco / w1
            nco = 1 + temp
        else:
            strobe = 0
            mu = mu1 # mu stays the same
            nco = temp
        
        if strobe == 0:
            eTime = 0
        else:
            k += 1
            xi = 0
            dxi = 0
            yi = 0
            dyi = 0
            # interpolate
            if (j >=1 and j < len(x) - 2):
                xi = cubicInterp(x[j - 1 : j + 2 + 1], mu)
                yi = cubicInterp(y[j - 1 : j + 2 + 1], mu)
                dxi = cubicInterp(dx[j - 1 : j + 2 + 1], mu)
                dyi = cubicInterp(dy[j - 1 : j + 2 + 1], mu)
            
            # rotate the symbol if necessary
            ptRot = np.array([xi, yi])
            ptRot = ptRot @ Gpo
            
            ##################################
            ## Begin freq offset pll
            ##################################
            # Evaluate the ML PED
            # frequency offset rotation matrix      
            Cfo = np.cos(-theta)
            Sfo = np.sin(-theta)
            
            # perform the PLL rotation
            xr = Cfo * ptRot[0] - Sfo * ptRot[1]
            yr = Sfo * ptRot[0] + Cfo * ptRot[1]
            
            decX.append(xr)
            decY.append(yr)

            dxi = Cfo * dxi - Sfo * dyi
            dyi = Sfo * dxi + Cfo * dyi
            
            # make a decision
            tmpPt = np.array([xr, yr])
            symbol = spatial.KDTree(LUT).query(tmpPt)[1]
                    
            a0 = LUT[symbol][0]
            a1 = LUT[symbol][1]
                
            ePhase = yr * a0 - xr * a1
            phErrorSig.append(ePhase)
            
            eTime = a0 * dxi + a1 * dyi
            timeErrorSig.append(eTime)
            muk.append(mu)
            
            # apply the loop filter
            k1Phaseout = ePhase * k1Phase
            k2PhaseOut = ePhase * k2Phase + k2PhaseOut
            vPhase = k1Phaseout + k2PhaseOut
            k0PhaseOut = vPhase * k0Phase
            theta = k0PhaseOut + theta
            phEstimate.append(theta)
            
            colDataX.append(a0)
            colDataY.append(a1)
            colDataSyms.append(symbol)
            ##################################
            ## End freq offset pll
            ##################################
        
            # when we reach the unique word, add the column to the image array
            if colIdx >= 229:
                bUwFound = False
                angleIdx = 0

                for i in range(len(angles)):
                    # allow for one error in the unique word
                    if (np.sum(np.isclose(colDataX[colIdx - uwSz + 1 : colIdx + 1], uwrotsyms[:, 0, i], 1e-1, 1).astype(int)) >= uwSz - 1 and \
                        np.sum(np.isclose(colDataY[colIdx - uwSz + 1 : colIdx + 1], uwrotsyms[:, 1, i], 1e-1, 1).astype(int)) >= uwSz - 1):
                        bUwFound = True
                        angleIdx = i

                        break
                
                # if we've found the unique word, we've found finished the column.
                # Rotate based on unique word phase offset, decide which symbols were
                # sent in this column and add the new column of symbols to the image 
                if bUwFound:
                    colDataX = colDataX[-230 :] # fix the column length for now (we know it's 230)
                    colDataY = colDataY[-230 :]
                    colDataSyms = colDataSyms[-230 :]
                    # this handles any multiples of 90 degree offsets
                    CsymRot = np.cos(angles[angleIdx])
                    SsymRot = np.sin(angles[angleIdx])
                    GsymRot = np.array([[CsymRot, -SsymRot],[SsymRot, CsymRot]])
                        
                    # If there's any rotation, rotate all the symbols in the column
                    if not np.isclose(angles[angleIdx], 0.0):
                        for i in range(len(colDataSyms)):
                            # rotate the current symbol by a multiple of 90 degrees
                            tmp = np.array([colDataX[i], colDataY[i]])
                            tmp = tmp @ GsymRot
                            
                            # make a decision on the symbol (find the closest LUT value to the
                            # tmp)
                            symbol = spatial.KDTree(LUT).query(tmp)[1]
                            
                            # update the symbol with the rotated value
                            colDataSyms[i] = symbol
                    
                    # append the column to the image
                    # (note that this appends the column as a row to the image, 
                    #  will transpose at the end to avoid thrashing. So it's
                    #  matlab contiguous until the end)
                    imgSyms.append(colDataSyms)
                    
                    # reset/update for the next column
                    colDataX = []
                    colDataY = []
                    colDataSyms = []
                    ncols += 1
                    colIdx = 0
                    continue
            
            # update the column index (the index in the column)
            colIdx += 1
            
        # compute timing loop filter output v(t)
        v1Time = k1Time * eTime
        v2Time = v2Time + k2Time * eTime
        vTime = v1Time + v2Time
        timeLfOut.append(vTime)
        
        # compute NCO input
        w = vTime + 1 / N
        nco1 = nco
        etak.append(temp)
        mu1 = mu
        w1 = w
    
    # need to transpose since we added the columns as rows 
    # (this was to keep it contiguous to avoid thrashing and
    # speed up the process)
    return np.transpose(np.array(imgSyms)), phErrorSig, phEstimate, timeLfOut, muk, decX, decY
      
    