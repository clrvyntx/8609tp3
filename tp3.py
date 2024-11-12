import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sgn

def autocorrelacion(x):
    return sgn.correlate(x,x,method='direct') / len(x)**2

def param_ar(x, p):
    rx = autocorrelacion(x)
    m = int(len(rx)/2)

    R = np.zeros((p,p))
    for i in range(0,p):
        R[i] = rx[m-i:m-i+p]

    r = rx[m+1:m+1+p]
    R = np.linalg.inv(R)

    a = np.matmul(R,r)
    g = rx[m] - np.dot(a,r)

    return np.append(1,-a), np.sqrt(g)

def generar_vocal(a, g, fs, t, f0):
    n = int(fs * t)
    d = int(fs / f0)
    l = int(t * f0)
    vocal = np.zeros(n)
    for i in range(0,l):
        vocal[d * i] = np.sqrt(fs / f0)
    return sgn.lfilter(g,a,vocal)

fs, audio = wav.read("./archivos/a.wav")
audio = np.array(audio,dtype=np.float64)
muestras = len(audio)
duracion = muestras / fs
t = np.linspace(0,duracion,muestras)

plt.figure(1)

plt.plot(t,audio)
plt.title('Señal de audio original')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()

p = 20
a, g = param_ar(audio,p)

f, sx = sgn.periodogram(audio,fs=fs,nfft=4096)
w, h = sgn.freqz(g,a,fs=fs)
pxx = np.abs(h)**2

plt.figure(2)

plt.plot(f,10 * np.log10(sx,where=sx>0))
plt.plot(w,10 * np.log10(pxx,where=pxx>0))

plt.title('Comparación entre periodograma del audio real y estimación de la PSD teórica')
plt.legend(['Periodograma','PSD teórica con parámetros estimados'])
plt.xlabel('Frecuencia [f]')
plt.ylabel('Amplitud [dB]')
plt.grid()

f1, pxx1 = sgn.welch(audio,fs=fs,window='hamming',nperseg=10,noverlap=5,nfft=4096)
f2, pxx2 = sgn.welch(audio,fs=fs,window='hamming',nperseg=100,noverlap=50,nfft=4096)
f3, pxx3 = sgn.welch(audio,fs=fs,window='hamming',nperseg=1000,noverlap=500,nfft=4096)

plt.figure(3)

plt.plot(f3,10 * np.log10(pxx3,where=pxx3>0))
plt.plot(f2,10 * np.log10(pxx2,where=pxx2>0))
plt.plot(f1,10 * np.log10(pxx1,where=pxx1>0))
plt.plot(w,10 * np.log10(pxx,where=pxx>0))

plt.title('Comparación entre método de Welch y estimación de la PSD teórica')
plt.legend(['Welch con M = 10','Welch con M = 100','Welch con M = 1000','PSD teórica con parámetros estimados'])
plt.xlabel('Frecuencia [f]')
plt.ylabel('Amplitud [dB]')
plt.grid()

vocal = generar_vocal(a,g,fs,0.5,100)
f4, pxx4 = sgn.welch(vocal,fs=fs,window='hamming',nperseg=100,noverlap=50,nfft=4096)
wav.write("./vocal.wav",fs,vocal.astype(audio.dtype))

plt.figure(4)

plt.plot(f4,10 * np.log10(pxx4,where=pxx4>0))
plt.plot(w,10 * np.log10(pxx,where=pxx>0))

plt.title('Comparación entre periodograma del audio sintetizado y estimación de la PSD teórica')
plt.legend(['Periodograma','PSD teórica con parámetros estimados'])
plt.xlabel('Frecuencia [f]')
plt.ylabel('Amplitud [dB]')
plt.grid()

plt.show()
