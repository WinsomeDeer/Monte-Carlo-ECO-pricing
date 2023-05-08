import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import scipy.stats as stats

S = 100
K = 98.31
vol = 0.102548
r = 0.01
t = 50
N = 10000

option_price = 4.86
T = ((datetime.date(2022, 4, 1) - datetime.date(2022, 1, 1)).days + 1)/365

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

C0 = np.exp(-r*T) * sum_Ct/N
sigma = np.sqrt((sum_Ct2 - sum_Ct**2/N) * np.exp(-r*T)/(N-1))
SE = sigma/np.sqrt(N)

x1 = np.linspace(C0 - 3*SE, C0 - SE, 100)
x2 = np.linspace(C0 - SE, C0 + SE, 100)
x3 = np.linspace(C0 + SE, C0 + 3*SE, 100)

s1 = stats.norm.pdf(x1, C0, SE)
s2 = stats.norm.pdf(x2, C0, SE)
s3 = stats.norm.pdf(x3, C0, SE)

plt.fill_between(x1, s1, color='tab:blue',label='> StDev')
plt.fill_between(x2, s2, color='cornflowerblue',label='1 StDev')
plt.fill_between(x3, s3, color='tab:blue')

plt.plot([C0, C0],[0, max(s2)*1.1], 'k', label = 'Theoretical Value')
plt.plot([option_price, option_price],[0, max(s2)*1.1], 'r', label = 'Market Value')
plt.ylabel("Probability")
plt.xlabel("Option Price")
plt.legend()
plt.show()
