import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sgn
import scipy.stats as stat

def suavizar_bordes(x, fade):
    m = len(x)
    fade = min(max(fade, 1), 50)

    n = 2 * int(fade / 100 * m) // 2
    v = np.hamming(n)

    fade_in = v[:n // 2]
    fade_out = v[n // 2:]

    return np.concatenate((fade_in, np.ones(m - n), fade_out)) * x

def autocorrelacion(x):
    return sgn.correlate(x,x,method='direct') / len(x)

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

def generar_fonema_sordo(a, g, fs, t):
    n = int(fs * t)
    fonema = stat.norm.rvs(loc=0,scale=1,size=n)
    return sgn.lfilter(g,a,fonema)

def generar_fonema_vocal(a, g, fs, t, f0):
    n = int(fs * t)
    d = int(fs / f0)
    l = int(t * f0)
    fonema = np.zeros(n)
    for i in range(0,l):
        fonema[d * i] = np.sqrt(fs / f0)
    return sgn.lfilter(g,a,fonema)

def concatenar_fonemas(archivo,fonemas,fs,fade):
    for i in range(0,len(fonemas)):
        fonemas[i] = suavizar_bordes(fonemas[i],fade)
    wav.write("./" + archivo,fs,np.concatenate(fonemas, axis=None).astype(np.int16))

def analisis_archivo(archivo,p,f0,tf):
    fs, audio = wav.read("./archivos/" + archivo)
    audio = np.array(audio,dtype=np.float64)
    muestras = len(audio)
    duracion = muestras / fs
    t = np.linspace(0,duracion,muestras)

    plt.figure(1)

    plt.plot(t,audio)
    plt.title('Señal de audio ' + archivo)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid()

    f, sx = sgn.periodogram(audio,fs=fs,nfft=4096)
    a, g = param_ar(audio,p)
    w, h = sgn.freqz(g,a,fs=fs)
    pxx = np.abs(h)**2 / muestras

    plt.figure(2)

    plt.plot(f,10 * np.log10(sx,where=sx>0))
    plt.plot(w,10 * np.log10(pxx,where=pxx>0))

    plt.title('Comparación entre periodograma y estimación de la PSD teórica del archivo ' + archivo)
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

    plt.title('Comparación entre método de Welch y estimación de la PSD teórica del archivo ' + archivo)
    plt.legend(['Welch con M = 10','Welch con M = 100','Welch con M = 1000','PSD teórica con parámetros estimados'])
    plt.xlabel('Frecuencia [f]')
    plt.ylabel('Amplitud [dB]')
    plt.grid()

    if(f0 == 0):
        fonema = generar_fonema_sordo(a,g,fs,tf)
    else:
        fonema = generar_fonema_vocal(a,g,fs,tf,f0)

    f4, pxx4 = sgn.welch(fonema,fs=fs,window='hamming',nperseg=100,noverlap=50,nfft=4096)
    wav.write("./" + archivo,fs,fonema.astype(np.int16))

    plt.figure(4)

    plt.plot(f4,10 * np.log10(pxx4,where=pxx4>0))
    plt.plot(w,10 * np.log10(pxx,where=pxx>0))

    plt.title('Comparación entre periodograma del audio sintetizado y estimación de la PSD teórica del archivo ' + archivo)
    plt.legend(['Periodograma','PSD teórica con parámetros estimados'])
    plt.xlabel('Frecuencia [f]')
    plt.ylabel('Amplitud [dB]')
    plt.grid()

    plt.show()

    return fonema, fs

a, fs = analisis_archivo("a.wav",20,100,0.5)
e, fs = analisis_archivo("e.wav",20,100,0.5)
f, fs = analisis_archivo("f.wav",20,0,0.5)
i, fs = analisis_archivo("i.wav",20,100,0.5)
j, fs = analisis_archivo("j.wav",20,0,0.5)
o, fs = analisis_archivo("o.wav",20,100,0.5)
s, fs = analisis_archivo("s.wav",20,0,0.5)
sh, fs = analisis_archivo("sh.wav",20,0,0.5)
u, fs = analisis_archivo("u.wav",20,100,0.5)

concatenar_fonemas("fonemas_concatenados.wav",[a,e,i,o,u,f,j,s,sh],fs,30)
