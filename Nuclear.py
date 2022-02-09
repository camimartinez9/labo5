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
plt.figure()
plt.plot(t1, v1)
plt.xlabel('Tiempo(s)')
plt.ylabel('Voltaje(V)')
plt.grid()
#Ruido del sistema sin fuente radiactiva
#%%
# plt.title('Voltaje en función del tiempo-Amplificador')
# plt.plot(t2, v2)
# plt.grid()
# plt.legend()
#%%
Picos=[]
# Tiempo_picos=[]
for i in range(2000): #Tomar 20 pantallas
    t1, v1 = osc.getWindow(1)
    # plt.plot(t2, v2)
    # plt.xlabel('Tiempo(s)')
    # plt.ylabel('Voltaje(V)')
    # # plt.figure(i)
    # plt.grid()
    #picos devuelve los lugares de la fila donde se encuentran efectivamente los picos
    picos,_=scipy.signal.find_peaks(v1, height=0.1)#,None,0)  
    #el guión bajo es para la otra info que da el findpeaks que no me importa
    # tiempo_picos=t1[picos]
    voltaje_picos=v1[picos]
    for j in voltaje_picos:            
            Picos.append(j)
    # Tiempo_picos.append(tiempo_picos)
    # cantidad_de_picos=len(picos)
    i=i+1
    print(i) #para ir viendo que se está haciendo el loop
# plt.figure()
# plt.plot(t1,v1)
# plt.xlabel('Tiempo(s)')
# plt.ylabel('Voltaje')
# plt.plot(Tiempo_picos[-1],Picos[-1],'o',color='r', label = 'Picos')
# plt.legend()
# plt.grid()
#Histograma
#%%

bineado = np.arange(0, 12, 0.16)
histos, bin_edges = np.histogram(Picos, bineado)

plt.plot(bin_edges[:-1], histos, 'o')

#plt.figure()
# plt.hist(Picos, bins=bineado)
plt.yscale('log')
plt.xlabel('Voltaje')
plt.ylabel('Cantidad de picos')
# intervalo=range(min(picos),max(picos)+1)
# plt.hist(cantidad_de_picos,intervalo)
