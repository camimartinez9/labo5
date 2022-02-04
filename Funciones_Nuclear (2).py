import matplotlib.pyplot as plt
import numpy as np
import nidaqmx
import math
import time
import scipy.signal as sp
import math


#para saber el ID de la placa conectada (DevX)
system = nidaqmx.system.System.local()
for device in system.devices:
    print(device)

#para setear (y preguntar) el modo y rango de un canal analógico
with nidaqmx.Task() as task:
    ai_channel = task.ai_channels.add_ai_voltage_chan("Dev2/ai1",max_val=10,min_val=-10)
    print(ai_channel.ai_term_cfg)    
    print(ai_channel.ai_max)
    print(ai_channel.ai_min)	
    
#%%
def medicion_corta(fs, tiempo_medicion): #tiempo en segundos
    #devuelve la data cruda
    cantidad_mediciones = tiempo_medicion*fs
    data = np.zeros(cantidad_mediciones*2)
    data[:] = np.nan
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
        task.ai_channels.add_ai_voltage_chan("Dev2/ai1", terminal_config = modo, max_val=10,min_val=-10)
        task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
        task.start()
        t0 = time.time()
        total = 0
        med = 0
        while med < cantidad_mediciones:
            time.sleep(0.1)
            datos_temp = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)  
            for i in range(len(datos_temp)):
                data[med+i] = datos_temp[i]
            total = total + len(datos_temp)
            t1 = time.time()
            med = med + len(datos_temp)
            print("%2.3fs %d %d %2.3f" % (t1-t0, len(datos_temp), total, total/(t1-t0))) 

        #saco los nan
    data = [value for value in data if not math.isnan(value)]
    #np.savetxt(filename+'.txt', np.transpose([data]))
    return data


#Plotear medicion corta

def medicion_picos(filename, fs, tiempo_medicion, altura_picos):
    #tiempo en segundos
    cantidad_mediciones = tiempo_medicion*fs
    data_peaks = np.zeros(cantidad_mediciones*2)
    data_peaks.fill(np.nan)
    index_peaks = np.arange(cantidad_mediciones*2, dtype=int)

    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
        task.ai_channels.add_ai_voltage_chan("Dev2/ai1", terminal_config = modo, max_val=10,min_val=-10)
        task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
        task.start()
        t0 = time.time()
        total = 0
        med = 0
        while med < cantidad_mediciones:
            time.sleep(0.05)  #intervalo de cada cuanto le pedimos tandas de datos al DAQ. Un time.sleep más grande que 0.1 puede llevar a que se llene el buffer
            datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)   
            
            datos_index_peaks,_ = sp.find_peaks(datos, height=altura_picos, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
#            for index, value in enumerate(datos_index_peaks):
#                j = int(med+index)
#                data_peaks[j] = datos[value]
#                index_peaks[j] = med+value
            X = len(datos_index_peaks)

            for index, value in enumerate(datos_index_peaks):
                i = int(med+index)
                if i<= len(data_peaks):
                    data_peaks[i] = datos[value]
                    index_peaks[i] = med+value
    
            total = total + len(datos)
            t1 = time.time()
            
            med = med + len(datos)
            print("%2.3fs %d %d %2.3f" % (t1-t0, len(datos), total, total/(t1-t0))) 

        #saco los nan
    data_peaks = [value for value in data_peaks if not math.isnan(value)]
    index_peaks = [value for value in index_peaks if not math.isnan(value)]
    time_peaks = [index/fs for index in index_peaks]
    #np.savetxt(filename+'.txt',np.transpose([data_peaks, time_peaks]), delimiter = ',')
    return data_peaks, time_peaks

def plot_histograma(data_peaks,tiempo_peaks, bines, escala):
  plt.figure()
  plt.xlabel('Tensión [V'), plt.ylabel('Intensidad')
  plt.hist(data_peaks, bins=bines)
  plt.yscale(escala)
  #plt.ylim(y1,y2)
  return 

def plot_medicionCorta(fs, data, altura_picos, x1,x2): #x1, x2 límites para plotear
  indices = np.arange(len(data))
  tiempo = [x/fs for x in indices]

  index_peaks, _ = sp.find_peaks(data, height=altura_picos, distance=None) 
  data_peaks = [data[i] for i in index_peaks]
  tiempo_peaks = [tiempo[i] for i in index_peaks]

  plt.figure(1,figsize=(20,5)), plt.clf()
  plt.xlabel('Tiempo [s]'), plt.ylabel('Tensión [V]')
  plt.plot(tiempo, data, '.-')
  plt.plot(tiempo_peaks, data_peaks, 'x', color='red')
  plt.hlines(altura_picos,min(tiempo),max(tiempo))
  #plt.xlim(x1, x2)

  bines = 500
  voltaje_positivo = [v for v in data if v>=0]
  plt.figure(2), plt.clf()
  plt.xlabel('Tensión [V]'), plt.ylabel('Intensidad')
  plt.hist(voltaje_positivo, bins=bines)
  plt.hist(altura_picos, bins= bines)
  plt.yscale('log')
  return

def medicion_continua(fs, tiempo_medicion):
    #tiempo en segundos
    picosPorSegundo = cuentaPicos(fs)
    cantidad_mediciones = tiempo_medicion*fs
    cantidad_picos = tiempo_medicion*picosPorSegundo*2
    data_peaks = np.zeros(cantidad_picos)
    data_peaks.fill(np.nan)
    index_peaks = np.zeros(cantidad_picos)
    index_peaks.fill(np.nan)

    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
        task.ai_channels.add_ai_voltage_chan("Dev2/ai1", terminal_config = modo,max_val=10,min_val=-10)
        task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
        task.start()
        t0 = time.time()
        tl = t0
        total = 0
        med = 0
        X = 0
        
        
        while med < cantidad_mediciones:
            time.sleep(0.005)
            datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)   
            
            t1fp = time.time() 
            datos_index_peaks,_ = sp.find_peaks(datos, height=0.1, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
            tfp = time.time() - t1fp
            
            for index, value in enumerate(datos_index_peaks):
                    data_peaks[X+index] = datos[value]
                    index_peaks[X+index] = X+value
    
            X = X + len(datos_index_peaks)

            total = total + len(datos)
            t1 = time.time() 
            
            med = med + len(datos)
            print("%2.3fs %d %d %2.3f %g" % (t1-t0, len(datos), total, total/(t1-t0),tfp)) 

        #saco los nan
    data_peaks = data_peaks[:X]
    index_peaks = index_peaks[:X]
    return data_peaks, index_peaks

#%% MEDIMOS

#Medicion corta
fs = 400000
tiempo_de_medicion = 10
filename = "CesioPrueba" 
data = medicion_corta(fs, tiempo_de_medicion)
plot_medicionCorta(fs, data, altura_picos=0.1, x1=0,x2=30)

#%%
#Medicion larga
tiempo_medicion = 5
altura_picos = 0.1

fs = 400000
tiempo_de_medicion = 21600
ini, fini = 10, 500
data, indices = medicion_continua(fs, tiempo_de_medicion)
plot_histograma(data,indices, bines=500, escala='log')


filename = "Cobalto4" 
np.savetxt(filename+'.txt',np.transpose([data, indices]), delimiter = ',')
