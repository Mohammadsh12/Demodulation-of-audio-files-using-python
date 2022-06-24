from ntpath import realpath
import wave 
from scipy.io import wavfile
import numpy as np 
import matplotlib.pyplot as plt 
from sympy import * 
def TimeFreq(y, Fs, BWrange):
  n = len(y) # length of the signal
  k = np.arange(n)
  T = n/Fs
  
  t = np.arange(0,n*Ts,Ts) # time vector

  frq = k/T # two sides frequency range
  fcen=frq[int(len(frq)/2)]
  frq_DS=frq-fcen
  frq_SS = frq[range(int(n/2))] # one side frequency range

  Y = np.fft.fft(y) # fft computing and normalization
  yinv= np.fft.ifft(Y).real # ifft computing and normalization
  Y_DS=np.roll(Y,int(n/2))
  Y_SS = Y[range(int(n/2))]
  return yinv


def plotTimeFreq(y, Fs, BWrange):
  n = len(y) # length of the signal
  k = np.arange(n)
  T = n/Fs
  
  t = np.arange(0,n*Ts,Ts) # time vector

  frq = k/T # two sides frequency range
  fcen=frq[int(len(frq)/2)]
  frq_DS=frq-fcen
  frq_SS = frq[range(int(n/2))] # one side frequency range

  Y = np.fft.fft(y) # fft computing and normalization
  yinv= np.fft.ifft(Y).real # ifft computing and normalization
  Y_DS=np.roll(Y,int(n/2))
  Y_SS = Y[range(int(n/2))]

  fcenIndex = (np.abs(frq_DS)).argmin()
  RangeIndex = (np.abs(frq_DS-BWrange)).argmin() - fcenIndex

  RangeIndexMin = fcenIndex-RangeIndex
  if RangeIndexMin < 0:
    RangeIndexMin = 0

  RangeIndexMax = fcenIndex+RangeIndex
  if RangeIndexMax > len(frq_DS)-1:
    RangeIndexMax = len(frq_DS)-1

  fig, ax = plt.subplots(2, 1,  figsize=(16, 6))
  ax[0].plot(t,y)
  ax[0].set_xlabel('Time')
  ax[0].set_ylabel('Amplitude')
  ax[1].set_xlabel('Freq (Hz)')
  ax[1].set_ylabel('|Y(freq)|')
  ax[1].plot(frq_DS[RangeIndexMin:RangeIndexMax],abs(Y_DS[RangeIndexMin:RangeIndexMax]),'r') # plotting the spectrum
  ax[1].set_xlabel('Freq (Hz)')
  ax[1].set_ylabel('|Y(freq)|')

  return yinv
def filter(y,Fs,B):
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    fcen=frq[int(len(frq)/2)]
    frq_DS=frq-fcen
    
    Y = np.fft.fft(y)
    fBWIndex = (np.abs(frq_DS - B)).argmin()
    B = frq_DS[fBWIndex]
    Y_DS=np.roll(Y,int(n/2))
    Mask_DS=np.ones(len(frq_DS))
    Yf_DS=np.copy(Y_DS)
    Bmax=frq_DS[len(frq_DS)-1]
    Bmin=0
    Bold=0
    Yf_DS=np.copy(Y_DS)
    for cnt in range(len(frq_DS)):
        if ~(((frq_DS[cnt])>-1*B) and ((frq_DS[cnt])<B)):
            Mask_DS[cnt]=0;
            #print(B,frq_DS[cnt],Yf_DS[cnt])
            Yf_DS[cnt]=Y_DS[cnt]*0;


    Yf=np.roll(Yf_DS,int(n/2))
    yinv= np.fft.ifft(Yf).real # ifft computing and normalization
    yinv=np.array(yinv)
    yinv_int=yinv.astype(np.int16)
    return yinv_int
    
  
filename = 'C:\\Users\\majd_\\Desktop\\LL.wav'
#2466 count
#2000 birds
#2266 athan
#2733 birds with noise
#3000 birds 
fc1 =2000
fc2=2266
fc3=2467
fc4=2733
fc5=3000
fcmax=270000
BW=9000
BWrange=11000

########-------------------
upsamplerate = int(fcmax/BW)


rate1, data1 = wavfile.read(filename)

ratemin=np.min([rate1])


Fs=ratemin*upsamplerate;
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,(len(data1)*Ts),Ts) # time vector
y=[float(x) for x in data1]

y1=y
TRIAL1 = plotTimeFreq(y1, Fs, BWrange)
carrier_signal1 = np.cos(2*np.pi*fc1*t)

output_signal1 = y1*carrier_signal1

output_signal_1=(output_signal1)

y2=y

carrier_signal2 = np.cos(2*np.pi*fc2*t)

output_signal2 = y2*carrier_signal2

output_signal_2=(output_signal2)

y3=y

carrier_signal3 = np.cos(2*np.pi*fc3*t)

output_signal3 = y3*carrier_signal3

output_signal_3=(output_signal3)

y4=y

carrier_signal4 = np.cos(2*np.pi*fc4*t)

output_signal4 = y4*carrier_signal4

output_signal_4=(output_signal4)
y5=y

carrier_signal5 = np.cos(2*np.pi*fc5*t)

output_signal5 = y5*carrier_signal5

output_signal_5=(output_signal5)


y1 = (output_signal_1)
y2 = (output_signal_2)
y3 = (output_signal_3)
y4 = (output_signal_4)
y5 = (output_signal_5)

yinv1 = TimeFreq(y1, Fs, BWrange)
yinv2 = TimeFreq(y2, Fs, BWrange)
yinv3 = TimeFreq(y3, Fs, BWrange)
yinv4 = TimeFreq(y4, Fs, BWrange)
yinv5 = TimeFreq(y5, Fs, BWrange)

yinv1_int=filter(y1,Fs,95)
rate = ratemin

filenameSave1='C:\\Users\\majd_\\Desktop\\LO1.wav'
rate=int(rate)
wavfile.write(filenameSave1, rate, yinv1_int)
yinv1 = plotTimeFreq(yinv1_int, Fs, BWrange)

yinv2_int=filter(y2,Fs,90)
filenameSave2='C:\\Users\\majd_\\Desktop\\LO2.wav'
wavfile.write(filenameSave2, rate, yinv2_int)
yinv2 = plotTimeFreq(yinv2_int, Fs, BWrange)

yinv3_int=filter(y3,Fs,70)
filenameSave3='C:\\Users\\majd_\\Desktop\\LO3.wav'
wavfile.write(filenameSave3, rate, yinv3_int)
yinv3 = plotTimeFreq(yinv3_int, Fs, BWrange)

yinv4_int=filter(y4,Fs,80)
filenameSave4='C:\\Users\\majd_\\Desktop\\LO4.wav'
wavfile.write(filenameSave4, rate, yinv4_int)
yinv4 = plotTimeFreq(yinv4_int, Fs, BWrange)

yinv5_int=filter(y5,Fs,78)
filenameSave5='C:\\Users\\majd_\\Desktop\\LO5.wav'

wavfile.write(filenameSave5, rate, yinv5_int)
yinv5 = plotTimeFreq(yinv5_int, Fs, BWrange)
plt.show()