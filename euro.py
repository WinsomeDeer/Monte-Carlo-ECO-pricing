import numpy as np
import matplotlib as plt
import pandas as pd
import datetime

S = 100
K = 98.31
vol = 0.102548
r = 0.01
t = 50
N = 10000

option_price = 4.86
T = (datetime(2022, 1, 1) - datetime(2022, 4, 1))/365

dt = T/N
vol_dt = vol*np.sqrt(dt)
mu_dt = (r - 0.5*vol**2) 
lnS = np.log(S)

sum_Ct = 0
sum_Ct2 = 0

for i in range(N):
    lnSt = lnS
    for j in range(t):
        lnSt = lnSt + mu_dt + vol_dt*np.random.normal()
    St = np.exp(lnSt)
    Ct = np.maximum(0, St - K)
    sum_Ct += Ct
    sum_Ct2 += Ct**2

C0 = np.exp(-r*T) * np.sum(Ct[-1])/N
sigma = np.sqrt((sum_Ct2 - sum_Ct**2 ) * np.exp(-r*T)/(N-1))
SE = sigma/np.sqrt(N)
