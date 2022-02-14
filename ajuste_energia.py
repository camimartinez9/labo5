# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:40:23 2022

@author: cami
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import scipy.signal as sp
import math
import os

#%%

#leo los txt para cada elemento
os.chdir (r'C:\Users\Pc\Desktop\mediciones nuclear\ba')
p = np.array([])
for i in range(12):
    file = "Bario"+f'med-{i}.txt'
    Datos = np.loadtxt(file, delimiter=",")
    Picos= Datos[:, 0]
    p = np.concatenate((p, Picos))
#%% Vemos histograma
bines = np.arange(0, 12, 0.16)
plt.figure(), plt.clf()
histos, bin_edges = np.histogram(p, bines)
plt.plot(bin_edges[:-1], histos, 'o')  
plt.xlabel('Tensión (V)')
plt.ylabel('Cuentas')
plt.title('Espectro')
plt.yscale('log')
#%%
import statsmodels.api as sm
volts=[2.07,2.88,3.7,6.72]
energias=[360,570,660,1280]
#Ajuste por caudrados minimos pesado con incetidumbres en y
errory=0.01 
w=1/errory
errorx= 2 
X = sm.add_constant(energias)
wls_model = sm.WLS(volts,X, weights=w)
results = wls_model.fit()
o,Pen=results.params
#intervalo de confianza para ordenada al origen y pendiente
oint,Rint=results.conf_int(alpha=0.05)
deltaP=(Rint[1]-Rint[0])/2
deltao= (oint[1]-oint[0])/2

print("Pendiente=(", Pen,"+/-",deltaP,") ") #esta pendiente es 3EM/L**3
print("ordenada al origen=(", o,"+/-",deltao,") N")

#calculo las bandas de confianza y predicción
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std

st, data, ss2 = summary_table(results, alpha=0.05)

fittedvalues = data[:, 2] #resultado de los valores ajustados
predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T #bandas de confianza
prstd, iv_l, iv_u = wls_prediction_std(results,alpha=0.05) #bandas de predicción con P>0.95

plt.figure()
plt.errorbar(energias,volts,xerr=errorx,yerr=errory,fmt=".b") #grafico valores medidos
#plt.grid('on');#agrega una grilla al grafico

plt.xlabel('');
plt.ylabel('');
plt.plot(energias, fittedvalues, '-', lw=1) #grafico de la recta de ajuste
plt.plot(energias, iv_l, 'r--', lw=2) #banda de predicción inferior
plt.plot(energias, iv_u, 'r--', lw=2) #banda de predicción superior
plt.plot(energias, predict_mean_ci_low, 'g--', lw=1) #banda de confianza inferior
plt.plot(energias, predict_mean_ci_upp, 'g--', lw=1) #banda de confianza superior
titulo=('')
plt.title(titulo)

#estadistica
residuos=volts-fittedvalues
chi2 = sum( residuos**2 )
print('R^2=',results.rsquared)