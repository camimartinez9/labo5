#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 12:25:53 2022

@author: publico
"""



import pyvisa as visa
import time
import numpy as np
from matplotlib import pyplot as plt
from osciloscope import Osciloscope
import scipy
from scipy import signal
osc=Osciloscope(0)



#%%
rm = visa.ResourceManager('@py')
res = rm.list_resources('@py')
#%%
t1, v1 = osc.getWindow(1)
# t2, v2 = osc.getWindow(2)

plt.plot(t1, v1)
plt.xlabel('Tiempo(s)')
plt.ylabel('Voltaje(V)')
plt.grid()
#Ruido del sistema sin fuente radiactiva
#%%
plt.title('Voltaje en función del tiempo-Amplificador')
plt.plot(t2, v2)
plt.grid()
plt.legend()
#%%
for i in range(20):
    t1, v1 = osc.getWindow(1)
    plt.plot(t2, v2)
    plt.xlabel('Tiempo(s)')
    plt.ylabel('Voltaje(V)')
    # plt.figure(i)
    plt.grid()
    picos=scipy.signal.find_peaks(v1, threshold=0.1, distance=20)[0]#,None,0)
    plt.plot(t1[picos], v1[picos], 'or')
    cantidad_de_picos=len(picos)
    i=i+1
    print(i)
intervalo=range(min(picos),max(picos)+1)
plt.hist(cantidad_de_picos,intervalo)