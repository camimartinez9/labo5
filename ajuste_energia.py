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

from scipy.optimize import curve_fit

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
volts=[2.07,2.88,3.7,5.44,6.72]
energias=[360,570,660,1063,1280]
#Ajuste por caudrados minimos pesado con incetidumbres en y
errory=0.04 
w=1/errory
X = sm.add_constant(energias)
wls_model = sm.WLS(volts,X, weights=w)
results = wls_model.fit()
o,Pen=results.params
#intervalo de confianza para ordenada al origen y pendiente
oint,Rint=results.conf_int(alpha=0.05)
deltaP=(Rint[1]-Rint[0])/2
deltao= (oint[1]-oint[0])/2

print("Pendiente=(", Pen,"+/-",deltaP,") ") 
print("ordenada al origen=(", o,"+/-",deltao,") ")

#calculo las bandas de confianza y predicción
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std

st, data, ss2 = summary_table(results, alpha=0.05)

fittedvalues = data[:, 2] #resultado de los valores ajustados
predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T #bandas de confianza
prstd, iv_l, iv_u = wls_prediction_std(results,alpha=0.05) #bandas de predicción con P>0.95

plt.figure()
plt.errorbar(energias,volts,yerr=errory,fmt=".r") #grafico valores medidos
#plt.grid('on');#agrega una grilla al grafico

plt.xlabel('Energía (keV)');
plt.ylabel('Voltaje (V)');
plt.plot(energias, fittedvalues, '-', lw=1,color='pink') #grafico de la recta de ajuste
# plt.plot(energias, iv_l, 'r--', lw=2) #banda de predicción inferior
# plt.plot(energias, iv_u, 'r--', lw=2) #banda de predicción superior
# plt.plot(energias, predict_mean_ci_low, 'g--', lw=1) #banda de confianza inferior
# plt.plot(energias, predict_mean_ci_upp, 'g--', lw=1) #banda de confianza superior
titulo=('Recta de calibración')
plt.title(titulo)
plt.grid()
#estadistica
residuos=volts-fittedvalues
chi2 = sum( residuos**2 )
print('R^2=',results.rsquared)
print(chi2)
#%% ajuste gaussiano fotopicos    
#pienso que tenemos 20 mediciones de 10 segundos para cada muestra con las mismas condiciones, las analizo por
#separado a priori
#leo los txt para cada elemento
os.chdir (r'C:\Users\Pc\Desktop\mediciones nuclear\bi')
p1 = np.array([])
for i in range(20):
    file1 = "Bismuto207"+f'med-{i}.txt'
    Datos1 = np.loadtxt(file1, delimiter=",")
    Picos1= Datos1[:, 0]
    p1 = np.concatenate((p1, Picos1))    
os.chdir (r'C:\Users\Pc\Desktop\mediciones nuclear\ba')

p2 = np.array([])
for i in range(12):
    file2 = "Bario"+f'med-{i}.txt'
    Datos2 = np.loadtxt(file2, delimiter=",")
    Picos2= Datos2[:, 0]
    p2 = np.concatenate((p2, Picos2))    
# for j in p1:
#     j=int(j)
#     for i in range(len(p1)):
#         p1[i]=j
p3=np.array([])
os.chdir (r'C:\Users\Pc\Desktop\mediciones nuclear\cs')
for i in range(20):
    file3 = "Cesio"+f'med-{i}.txt'
    Datos3 = np.loadtxt(file3, delimiter=",")
    Picos3= Datos3[:, 0]
    p3 = np.concatenate((p3, Picos3)) 
p3_picos=[]
os.chdir (r'C:\Users\Pc\Desktop\mediciones nuclear\na')
p4=np.array([])
for i in range(20):
    file4 = "Cesio"+f'med-{i}.txt'
    Datos4 = np.loadtxt(file4, delimiter=",")
    Picos4= Datos4[:, 0]
    p4 = np.concatenate((p4, Picos4))              
histos4, bin_edges4 = np.histogram(p4, bines)
plt.plot(bin_edges4[:-1], histos4, 'o') 
histos1, bin_edges1 = np.histogram(p1, bines)
histos2, bin_edges2 = np.histogram(p2, bines)
histos3, bin_edges3 = np.histogram(p3, bines)
plt.figure()
plt.plot(bin_edges1[:-1], histos1, 'o')  
plt.plot(bin_edges2[:-1], histos2, 'o')
plt.plot(bin_edges3[:-1], histos3, 'o')    
#%%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy
# Generate some data for this 
# demonstration.
p_1_fotopico_1=[]
for i in p1:
    if i<3.7 and i>2.2:
        p_1_fotopico_1.append(i)
p1_fotopico_2=[]
for i in p1:
    if i<6.072 and i>4.5:
        p1_fotopico_2.append(i)
p2_fotopico=[]
for i in p2:
    if i>0.9 and i<3.4:
        p2_fotopico.append(i)
    
data = p_1_fotopico_1
  
# Fit a normal distribution to
# the data:
# mean and standard deviation

# Plot the histogram.
bines=np.arange(2.2,3.7,0.16)
histos1, bin_edges = np.histogram(data, bines)
plt.plot(bin_edges[:-1], histos1, 'o')  
  
# Plot the PDF.
mu, sigma = scipy.stats.norm.fit(data)
bins=bin_edges[:-1]
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line)
    
 #%% volts a energia
def volts_a_energia(volts):
    a= 0.0050356215047
    b=0.200980
    energia=(volts-b)/a
    return energia
picos_en_energia=volts_a_energia(p1)
p_1_fotopico_1_array=np.array(p_1_fotopico_1)
picos_en_energia_region=volts_a_energia(p_1_fotopico_1_array)
  #%%
plt.figure()
data=  p2_fotopico
_, bins, _ = plt.hist(data, 100, density=2, alpha=0.5)

mu, sigma = scipy.stats.norm.fit(data)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line,label='Gaussiana')
plt.xlabel('Voltaje(V)')

plt.legend()
    


#%% analizo ventana a ventana
os.chdir (r'C:\Users\Pc\Desktop\mediciones nuclear\bi')
archivo = "Bismuto207"+f'med-{1}.txt'
# plt.figure()

Dato_bis = np.loadtxt(archivo, delimiter=",")
Picos_una_ventana_bis= Dato_bis[:, 0]

# bines = np.arange(0, 12, 0.16)
# histos, bin_edges = np.histogram(Picos_una_ventana_bis, bines)
# plt.plot(bin_edges[:-1], histos, 'o')     
# plt.show() 
#%%
# from scipy.stats import poisson
# k = Picos_una_ventana_bis
# pmf = poisson.pmf(k, mu=7)
# pmf = np.round(pmf, 5)
# plt.plot(k, pmf, marker='o')
# plt.xlabel('k')
# plt.ylabel('Probability')
# plt.figure()
# plt.show()
def poisson(n,l): #n eventos, numero de cuentas
    p=((l**n)*(np.e**-l))/(np.math.factorial(n))
    return p
# Indice_enteros=[]
# for j in Indice_picos:
#     j=int(j)
#     Indice_enteros.append(j)
# res,cov=curve_fit(poisson,Indice_enteros) 

# histos3, bin_edges = np.histogram(Picos_una_ventana_bis, bines)
# plt.plot(bin_edges[:-1], histos3, 'o') 
# histos3_mejor=[]
# for i in histos3:
#     if i<3000:
#         histos3_mejor.append(i)
# plt.hist(histos3_mejor,bins=20)
# plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson



data=Picos_una_ventana_bis

bins = np.arange(12) - 0.5 
entries, bin_edges, patches = plt.hist(data, bins=bins, density=True)


bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])


def fit_function(k, lamb):
    '''poisson function, parameter lamb is the fit parameter'''
    return poisson.pmf(k, lamb)


parameters, cov_matrix = curve_fit(fit_function, bin_middles, entries)


x_plot = np.arange(0, 12)

plt.plot(
    x_plot,
    fit_function(x_plot, *parameters),
    # marker='o', linestyle='',
    label='Poisson',
)
plt.legend()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import factorial
from scipy import stats


def poisson(k, lamb):
    """poisson pdf, parameter lamb is the fit parameter"""
    return (lamb**k/factorial(k)) * np.exp(-lamb)


def negative_log_likelihood(params, data):
    """
    The negative log-Likelihood-Function
    """

    lnl = - np.sum(np.log(poisson(data, params[0])))
    return lnl

def negative_log_likelihood(params, data):
    ''' better alternative using scipy '''
    return -stats.poisson.logpmf(data, params[0]).sum()


# get poisson deviated random numbers
data = p1

# minimize the negative log-Likelihood

result = minimize(negative_log_likelihood,  # function to minimize
                  x0=np.ones(1),            # start value
                  args=(data,),             # additional arguments for function
                  method='Powell',          # minimization method, see docs
                  )
# result is a scipy optimize result object, the fit parameters 
# are stored in result.x
print(result)

# plot poisson-distribution with fitted parameter
x_plot = np.arange(0, 12)

plt.plot(
    x_plot,
    stats.poisson.pmf(x_plot, *parameters),
    # marker='o', linestyle='',
    label='Fit result',
)
plt.legend()
plt.show()