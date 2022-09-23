import numpy as np

def cubicInterp(x, mu):
    xmp2 = x[3]
    xmp1 = x[2]
    xm = x[1]
    xmm1 = x[0]
    
    v3 = 1/6 * xmp2 - 1/2 * xmp1 + 1/2 * xm - 1/6 * xmm1
    v2 =   0 * xmp2 + 1/2 * xmp1 -   1 * xm + 1/2 * xmm1
    v1 = -1/6 * xmp2 + 1 * xmp1 - 1/2 * xm - 1/3 * xmm1
    v0 = xm
    
    return ((v3 * mu + v2) * mu + v1) * mu + v0
