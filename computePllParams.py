def computePllParams(gamma, Bn, T, k0, kp):
    k1 = (4 * gamma) * ((Bn * T) / (gamma + 1/ (4 * gamma))) / ((1 + 2 * gamma * (Bn * T / (gamma + 1 / (4 * gamma))) + (Bn * T / (gamma + 1 / (4 * gamma))**2)) * (k0 * kp))
    k2 = (4) * ((Bn * T) / (gamma + 1/ (4 * gamma)))**2 / ((1 + 2 * gamma * (Bn * T / (gamma + 1 / (4 * gamma))) + (Bn * T / (gamma + 1 / (4 * gamma))**2)) * (k0 * kp))
    return k1, k2

