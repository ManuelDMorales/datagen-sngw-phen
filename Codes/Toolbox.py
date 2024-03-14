# ---------------------------------------------------------------
# ------> Gravitational Wave Data Analysis Toolbox
# ---------------------------------------------------------------

# ------> Import libraries

# Files/folders management
import h5py
import fsspec

# Data analysis
import numpy as np 

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# Files/folders management
import re, os, glob, sys

# To read csv files
import csv

# Scientific computing
from scipy import interpolate
from scipy import signal

from scipy.signal import get_window
from scipy.signal import freqz
from scipy.signal import detrend

import scipy.fftpack as fftpack
import scipy as sp

# Garbage collector
import gc

# GW standard tools
import pycbc
import pycbc.psd

# Other useful tools
from collections import Counter
import itertools

# ---------------------------------------------------------------
# ------> Data downloader from GW Open Science Center
# ---------------------------------------------------------------

def files_from_url(url_detector):
  
    """
    Function to download a strain data noise segment (time series)
    and its metadata, based on its URL on the GW Open Science
    Center: https://gwosc.org/data.
    
    INPUT:
            url_detector -> hdf5 file's URL
            
    OUTPUT:
            strain   -> LIGO/Virgo detector strain data
            gpsStart -> Data segment's GPS start
            Duration -> Data segment's duration (in seconds)
            ts       -> Data segment's sampling time
    """

    # Open remote file in binary read mode
    remote_f = fsspec.open(url_detector, mode="rb")

    if hasattr(remote_f, "open"):
        remote_f = remote_f.open()

    File = h5py.File(remote_f)

    print("File Keys:", File.keys())

    # Extract strain and sampling time
    strain = File['strain']['Strain'][()]
    ts = File['strain']['Strain'].attrs['Xspacing']

    # Extract gpsStart, duration
    meta = File['meta']
    gpsStart = meta['GPSstart'][()]
    duration = meta['Duration'][()]

    return strain, gpsStart, duration, ts

# ---------------------------------------------------------------
# ------> Waveform filename info extractor
# ---------------------------------------------------------------

def extract_WFinfo(file_path):
    
    """
    Function to extract Phenomenological waveforms' information
    from their filenames
    
    INPUT:
            file_path -> Waveform file path (string)
            
    OUTPUT:
            slope  -> HFF slope
            f0     -> HFF start frequency
            f1     -> HFF end frequency 
    """
    
    # Search patterns
    pat_1 = re.search("Slope", file_path)
    pat_2 = re.search("_f0", file_path)
    pat_3 = re.search("_f1", file_path)
    pat_4 = re.search("\).csv", file_path)

    # Define indexes and save Slope value
    index_slope_1 = pat_1.span()[1]
    index_slope_2 = pat_2.span()[0]
    slope = file_path[index_slope_1:index_slope_2]

    # Define indexes and save f0 value
    index_f0_1 = pat_2.span()[0]+4
    index_f0_2 = pat_3.span()[0]-1
    f0= file_path[index_f0_1:index_f0_2]

    # Define indexes and save f2 value
    index_f1_1 = pat_3.span()[0]+4
    index_f1_2 = pat_4.span()[0]
    f1 = file_path[index_f1_1:index_f1_2]
    
    return slope, f0, f1

# ---------------------------------------------------------------
# ------> Waveform rescaling
# ---------------------------------------------------------------

def rescale_gw(h, d):
        
    """
    Gravitational waveform rescaler in function of distance
    
    INPUT:
            h -> Strain array
            d -> Distance from the emitting source (in Kpc)
            
    OUTPUT:
            h1 -> Rescaled strain array
    """  
    
    h1 = h * 10.0 / d
    
    return h1

# ---------------------------------------------------------------
# ------> Waveform resampling
# ---------------------------------------------------------------

def resample_gw(t, h, f):
        
    """
    Gravitational waveform resampler
    
    INPUT:
            t  -> Time array
            h  -> Strain array
            fs -> Output sampling frequency
            
    OUTPUT:
            t1 -> Resampled time array
            h1 -> Resampled strain array
    """
    
    # Check
    if len(t)!=len(h):
        print("Error: t and h need to have equal sizes")
        return 0
    
    # Define new time with f
    t1 = np.arange(t[0],t[-1],1.0/f)
    
    
    # Interpolation
    tck = interpolate.splrep(t,h,s=0.0)
#   print(tck[0])
#   print(tck[0])

    # Check duplicate values in time
#   dup = [item for item, count in Counter(tck[0]).items() if count > 1]

    # Remove duplicate values if necessary
    
#   if len(dup) >= 1:
        
#        loc_dup_rm = []
        
#        for i in range(len(dup)):
#            loc_dup_rm.append(np.where(tck[0]==dup[i])[0])
        
#        loc_rm = list(itertools.chain.from_iterable(loc_dup_rm))
        
#        t_cropped = np.delete(tck[0], loc_rm)
#        s_cropped = np.delete(tck[1], loc_rm)
        
#        tck = (t_cropped, s_cropped)
#        print(tck[0])
#        print(tck[1])
        
#   else:
        
#        pass
        
    h1  = interpolate.splev(t1,tck,der=0)
    
    return t1, h1
    #return tck
    
# ---------------------------------------------------------------
# ------> Waveform removal of edges
# ---------------------------------------------------------------

def remove_gw_edges(t, h, tL_lim, tR_lim):
    
    """
    Gravitational waveform edges' removal.
    Samples for t<tL_lim and t>t_lim are removed.
    
    INPUT:
            t      -> Time array
            h      -> Strain array
            tL_lim -> Left time limit (in seconds)
            tR_lim -> Right time limit (in seconds)
            
    OUTPUT:
            t1    -> Cropped time array
            h1    -> Cropped strain array
    """  
    
    # Compute number of samples in h
    n = len(h)

    # Compute sampling frequency
    ts = t[1]-t[0]
    
    # Compute limits in sample units
    nL_lim = int( (tL_lim-t[0]) / ts )
    print("First sample of cropped arrays =", nL_lim)
    
    nR_lim = int( (tR_lim-t[0]) / ts )
    print("Last sample of cropped arrays =", nR_lim)

    # Remove edges of strain array
    t1 = t[nL_lim:nR_lim+1]
    h1 = h[nL_lim:nR_lim+1]
    
    return t1, h1


# ---------------------------------------------------------------
# ------> Fast Fourier Transform (FFT) - Version 1
# ---------------------------------------------------------------

def fft_signal_v1(h, fs , nfft):
        
    """
    Function to compute FFT's values, running the absolute
    normalized frequencies (x axis) from 0*fs to nfft/2*fs.
    This is a single-side FFT with double magnitudes, where the
    right end point is the "folding frequency" (Nyquist frequency)
    Nyquist=fs/2. This version of FFT allow US to locate absolute
    frequencies of the signal and, therefore, to compute its SNR.
    
    INPUT:
            h    -> Signal
            fs   -> Sampling frequency
            nfft -> FFT size
            
    OUTPUT:
            f    -> FFT's frequency bins
            h_f  -> Signal's FFS
    """
    
    # Compute discrete Fourier transform
    h_f = (1/fs)*sp.fft.fft(h, nfft)
    f = sp.fft.fftfreq(nfft, 1/fs)
    
    # Shift frequencies
    h_f =  sp.fft.fftshift(h_f)
    f   =  sp.fft.fftshift(f)

    # The one sided fft (real data)
    f      = f[int(nfft/2)+1:nfft-1]
    h_f    = 2.0*h_f[int(nfft/2)+1:nfft-1]
        
    return f, h_f

# ---------------------------------------------------------------
# ------> Fast Fourier Transform (FFT) - Version 2
# ---------------------------------------------------------------

def fft_signal_v2(h, fs, nfft):
        
    """
    Function to compute FFT's raw values, running the frequencies
    (x axis) from 0*fs to 1*fs, segmented in nfft bins. This is a
    single-side FFT, where the right end point is twice the
    "folding frequency" (Nyquist frequency) 2*Nyquist=2*(fs/2).
    From this version of the FFT we cannot locate frequencies of
    the signal, but it is still valid, and easy to implement, for
    data whitening.
    
    INPUT:
            h    -> Signal
            fs   -> Sampling frequency
            nfft -> FFT size
            
    OUTPUT:
            f    -> FFT's frequency bins
            fft  -> Signal's FFS
    """

    f = fs * np.linspace(0, 1, nfft)
    fft = sp.fft.fft(h)

    return f, fft

# ---------------------------------------------------------------
# ------> Power Spectral Density with SciPy library
# ---------------------------------------------------------------

def PSD(h, fs, nperseg, sides):
        
    """
    Power spectral density estimator through Welch's method.
    
    INPUT:
            h       -> Signal
            fs      -> Sampling frequency
            nperseg -> Length of each Welch segment in sample units
            sides   -> For single (1), or double (2) sided-PSD
            
    OUTPUT:
            freq    -> FFT's frequency bins
            psd     -> Signal's FFS
    """
        
    # Compute the PSD
    # We mantain the following default settings: window='hann'
    
    nfft = nperseg
    # NFFT length = Welch window length
    # Remark: This FFT is for computing the PSD of each Welch window
    
    # nfft = int(len(h)/2)
    # nfft = int(len(h))
    # nfft = 100*nperseg
    
    if sides == 1:
        freq,psd = signal.welch(h, fs=fs, nperseg=nperseg, noverlap=nperseg//2, nfft=nfft, return_onesided=True)
    elif sides == 2:
        freq,psd = signal.welch(h, fs=fs, nperseg=nperseg, noverlap=nperseg//2, nfft=nfft, return_onesided=False)
    else:
        print("ERROR: \"sides\" input in PSD function only accepts 1 or 2 values")
        
    return freq, psd

# ---------------------------------------------------------------
# ------> Signal-to-noise ratio (SNR) estimator with SciPy
# ---------------------------------------------------------------

def SNR(h, t, fpsd, psd, doplots):
        
    """
    Signal-to-noise ratio estimator with SciPy library.
    
    INPUT:
            h       -> Signal array
            t       -> Time array
            fpsd    -> PSD frequency array
            psd     -> PSD estimation array
            doplots -> Do plots for checks (0: no, 1: yes)
            
    OUTPUT:
            SNR      -> Signal-to-noise ratio value
    """
    
    fs = 1/(t[1]-t[0])
    
    # Compute the nfft/2-points FFT of the strain data
    f1, fft = fft_signal_v1(h, fs, len(t))
    
    factor = 1.0

    fft = abs(fft)                                   
    fft = fft*factor
        
    if doplots:
        
        print("****** Check: Plot FFT ******")
        
        plt.figure(2, figsize=(8,5))
        plt.plot(f1, fft, label='FFT')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(left=100)
        plt.title("Fast Fourier Transform", fontsize=18)
        plt.ylabel("FFT magnitude", fontsize=13)
        plt.xlabel("Frequency (Hz)", fontsize=13)
        plt.legend()
        plt.show()
        plt.figure(2).clear()
        gc.collect()
        print("")
        print("Length of the one-sided FFT: ", len(f1))
        print("")
        #print(len(psd))
    
    else:
        pass
    
    # Interpolate PSD to have same dim with FFT
    tck = interpolate.splrep(fpsd,psd,s=0)
    psd = interpolate.splev(f1,tck,der=0)
    
    # Differential of the frequency
    df = f1[2] - f1[1]                               

    if doplots:
        
        print("****** Check: Plot Interpolated PSD ******")
        
        plt.figure(3, figsize=(7.75,5))
        plt.plot(f1, psd, label='PSD')
        #plt.xlim(left=9)
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Nfft points double sided PSD raw", fontsize=18)
        plt.ylabel("Amplitude Spectral Density", fontsize=13)
        plt.xlabel("Frequency (Hz)", fontsize=13)
        #plt.legend()
        plt.show()
        plt.figure(3).clear()
        gc.collect()
        print("")
    
    else:
        pass
    
    #print(f"PSD: {psd}")
    
    # Calculate the integral from the doc:
    SNRsq =  np.sum(((fft**2)*df)/psd)
    
    # Just a check
    print("+/- SNR**2 value:", SNRsq)

    # Calculate the SNR
    SNR = np.sqrt(abs(SNRsq))
    
    return SNR

# ---------------------------------------------------------------
# ------> Signal-to-noise ratio (SNR) estimator with PyCBC
# ---------------------------------------------------------------

def SNR_PyCBC(strain_wf, strain_n, fs, doplots):
        
    """
    Signal-to-noise ratio estimator with PyCBC library.
    
    INPUT:
            strain_wf -> Waveform strain
            strain_n  -> Noise strain
            doplots   -> Do plots for checks (0: no, 1: yes)
            
    OUTPUT:
            SNR       -> Signal-to-noise ratio value
    """
    
    # Sampling time
    ts = 1/fs
        
    # Convert numpy arrays to time series
    Noise =  pycbc.types.TimeSeries(strain_n, delta_t = ts)
    Waveform = pycbc.types.TimeSeries(strain_wf, delta_t = ts)

    # Length of each Welch segment
    seg_len = int(4*fs)   # in sample units

    # Overlap between Welch segments in seconds
    seg_stride = int(2*fs)  # in sample units

    # Welch's Power Spectral Density estimate
    psd = pycbc.psd.welch(Noise, seg_len=seg_len, seg_stride=seg_stride)
    psd = pycbc.psd.interpolate(psd, 1.0/len(Waveform)*fs)

    if doplots:
        
        print("****** Check: Plot Interpolated PSD ******")

        # Extract frequencies from PSD
        fpsd = psd.sample_frequencies
        
        plt.figure(3, figsize=(7.75,5))
        plt.plot(fpsd, psd, label='PSD')
        #plt.xlim(left=9)
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Nfft points double sided PSD raw", fontsize=18)
        plt.ylabel("Power Spectral Density", fontsize=13)
        plt.xlabel("Frequency (Hz)", fontsize=13)
        #plt.legend()
        plt.show()
        plt.figure(3).clear()
        gc.collect()
        print("")
    
    else:
        pass
        
    # Compute SNR values
    SNR = pycbc.filter.sigma(Waveform, psd, low_frequency_cutoff=1.0)

    return SNR

# ---------------------------------------------------------------
# ------> Whitening with SciPy library
# ---------------------------------------------------------------

    """
    Function to apply whitening to strain data,
    using the SciPy library.
    
    INPUT:
            h        -> Strain data (1-D array)
            fs       -> Strain data sampling frequency
            nperseg  -> Duration of each Welch segments (in sample units)
            doplots  -> Do plots for checks (0: no, 1: yes)
            
    OUTPUT:
            s_white_t  -> Whitened strain data
    """

def whitening(h, fs, nperseg, nfft, doplots):
    # h       -> signal
    # fs      -> sampling frequency
    # nperseg -> length of each Welch segment in sample units
    # nfft    -> FFT size in sample units
    
    # ------> Compute PSD based on input Welch length
    
    # Double-sided power spectral density of the raw strain data
    fpsd_raw, psd_raw = PSD(h, fs, nperseg, 2)

    # Sort arrays, increasing frequencies
    ind_raw = np.argsort(fpsd_raw)
    fpsd_raw = np.sort(fpsd_raw)
    psd_raw = psd_raw[ind_raw]
    
    print("****** Check: Dimensions  (in sample units) ******")
    print("")
    print("Input PSD raw:", len(psd_raw))
    print("Input PSD freq bins:", len(fpsd_raw))
    print("")
    
    # ------> Interpolate PSD to increase samples to nfft
    
    # nfft/2+1 points single-sided PSD of the raw strain data
    fpsd_ss = fs * np.arange(nfft//2 + 1) / nfft
    psd_ss = interpolate.interp1d(fpsd_raw, psd_raw, kind='linear',\
                                  bounds_error=False,\
                                  fill_value="extrapolate")(np.abs(fpsd_ss))
    
    # nfft-points double-sided PSD of the raw strain data
    fpsd_ds = fs * np.linspace(0,1,nfft)
    psd_ds = np.concatenate((psd_ss, np.flipud(psd_ss[1:-1])))
    
    print("Interpolated PSD raw:", len(psd_ds))
    print("Interpolated PSD freq bins:", len(fpsd_ds))
    print("")
    
    # ------> Check: Interpolated PSD of the raw strain data
    
    if doplots:
        
        #mpl.rcParams['agg.path.chunksize'] = 10000
        print("****** Check: Plot Interpolated PSD ******")
        
        plt.figure(1, figsize=(7.75,5))
        plt.plot(fpsd_ds, psd_ds, label='ASD')
        #plt.xlim(left=9)
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Nfft points double sided PSD raw", fontsize=18)
        plt.ylabel("Amplitude Spectral Density", fontsize=13)
        plt.xlabel("Frequency (Hz)", fontsize=13)
        plt.legend()
        plt.show()
        plt.figure(1).clear()
        gc.collect()
        print("")
    
    else:
        pass
    
    
    # ------> Compute Fast Fourier Transform
    
    # Windowed of strain data
    # window = get_window('hann', len(h))
    # h = h * window
    
    # nfft-points fft of the raw strain data
    f, fft = fft_signal_v2(h, fs, nfft)
    
    if doplots:
        
        #mpl.rcParams['agg.path.chunksize'] = 10000
        print("****** Check: Plot FFT ******")
        
        plt.figure(2, figsize=(8,5))
        #plt.plot(f, abs(fft), label='FFT of windowed strain')
        plt.plot(f, abs(fft), label='FFT of raw strain data')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Fast Fourier Transform", fontsize=18)
        plt.ylabel("FFT magnitude", fontsize=13)
        plt.xlabel("Frequency (Hz)", fontsize=13)
        plt.legend()
        plt.show()
        plt.figure(2).clear()
        gc.collect()
        print("")
    
    else:
        pass

    # ------> Compute whitening
    
    # Amplitude Spectral Density
    asd_raw = np.sqrt(psd_ds) 
    
    # Frequency-domain whitened strain signal
    s_white_f = np.divide(fft, asd_raw)
    
    # Scale frequency-domain whitened strain
    ind = np.where(np.logical_and(fpsd_raw>=100, fpsd_raw<=200))
    sf = np.mean(asd_raw[ind]) 
    s_white_f = sf*s_white_f
    
    # Time-domain whitened strain signal
    s_white_t = sp.fft.ifft(s_white_f)
    
    # Magnitude of whitened strain signal
    s_white_t = abs(s_white_t)

    if doplots:
        
        # ------> CHECK: Comparison ASD raw / ASD whitened
    
        # **** Raw strain data ASD ****
        
        # Double-sided power spectral density of the raw strain data
        fpsd_raw, psd_raw = PSD(h, fs, nperseg, 2)

        # Sort arrays, increasing frequencies
        ind_raw = np.argsort(fpsd_raw)
        fpsd_raw = np.sort(fpsd_raw)
        psd_raw = psd_raw[ind_raw]
        
        # Amplitude Spectral Density
        asd_raw = np.sqrt(psd_raw)
        
        # **** Whitened strain data ASD ****
        
        # Double-sided power spectral density of the whitened strain data
        fpsd_white, psd_white = PSD(s_white_t, fs, nperseg, 2)
        
        # Sort arrays according to increasing frequencies
        ind_white = np.argsort(fpsd_white)
        fpsd_white = np.sort(fpsd_white)
        psd_white = psd_white[ind_white]
        
        # Amplitude Spectral Density of whitened strain data
        asd_white = np.sqrt(psd_white)
        
        # **** Plot ASD curves ****
        
        #mpl.rcParams['agg.path.chunksize'] = 10000
        print("****** Check: Plot FFT ******")
        
        plt.figure(3, figsize=(7.75,5))
        plt.plot(fpsd_raw, asd_raw, label='ASD of raw data')
        plt.plot(fpsd_white, asd_white, label='ASD of whitened data')
        #plt.plot(fpsd_filt, asd_filt, label='ASD of filtered data')
        #plt.xlim(left=9)
        plt.xscale('log')
        plt.yscale('log')
        plt.title("ASD", fontsize=18)
        plt.ylabel("Amplitude Spectral Density", fontsize=13)
        plt.xlabel("frequency", fontsize=13)
        plt.legend()
        plt.show()
        plt.figure(3).clear()
        gc.collect()
    
    else:
        pass
        
    return s_white_t

# ---------------------------------------------------------------
# ------> Whitening with PyCBC library
# ---------------------------------------------------------------

def whitening_PyCBC(s, w_seg, cut, ts, doplots):

    """
    Function to apply whitening to strain data,
    using the standard LIGO PyCBC library.
    
    INPUT:
            s        -> Strain data (1-D array)
            w_seg    -> Duration of each Welch segments (in seconds)
            cut      -> Cut's length (in seconds) to remove edge artifacts
                        in whitened strain data:
                        (cut/2) at left edge, (cut/2) at right edge
            ts       -> Strain data sampling time
            doplots  -> Do plots for checks (0: no, 1: yes)
            
    OUTPUT:
            s_white_t  -> Whitened strain data
            psd        -> PSD of raw strain data
            fpsd       -> PSD frequencies
            psd_white  -> PSD of whitened strain data
            fpsd_white -> PSD_white's frequencies
    """

    # Convert strain numpy array to timeseries
    s = pycbc.types.timeseries.TimeSeries(s, ts)

    # Apply whitening
    s_white_t = s.whiten(w_seg, cut)

    # Recover PSD of raw strain data

    seg_len    = int(w_seg/ts)
    seg_stride = int(seg_len/2)
    
    psd = pycbc.psd.welch(s, seg_len=seg_len, seg_stride=seg_stride )
    fpsd = psd.sample_frequencies

    # Scale whitened signal: option 1
    #sc = min(psd**0.5)
    #s_white_t = s_white_t * sc
        
    # Scale whitened signal: option 2
    ind = np.where(np.logical_and(fpsd>=100, fpsd<=200))
    sc = np.mean(np.sqrt(psd[ind])) 
    s_white_t = s_white_t * sc

    # Estimate Whitened PSD
    psd_white = pycbc.psd.welch(s_white_t, seg_len=seg_len, seg_stride=seg_stride)
    fpsd_white = psd_white.sample_frequencies
        
    if doplots:
        
        #mpl.rcParams['agg.path.chunksize'] = 10000
        print("****** Check: Plot ASDs ******")
        
        plt.figure(1, figsize=(7.75,5))
        plt.plot(fpsd, np.sqrt(psd), label='ASD of raw data')
        plt.plot(fpsd_white, np.sqrt(psd_white), label='ASD of whitened data')
        #plt.xlim(left=9)
        plt.xscale('log')
        plt.yscale('log')
        plt.title("PSD", fontsize=18)
        plt.ylabel("Power Spectral Density", fontsize=13)
        plt.xlabel("Frequency (Hz)", fontsize=13)
        plt.legend()
        plt.show()
        plt.figure(1).clear()
        gc.collect()
        print("")
    
    else:
        pass

    return s_white_t, psd, fpsd, psd_white, fpsd_white


# ---------------------------------------------------------------
# ------> Band pass filter with SciPy library
# ---------------------------------------------------------------

def filt_bandpass(h, fs, n, band, doplots):

    """
    Function that applies a band pass filter to strain data,
    using the SciPy library.
    
    INPUT:
            h        -> Strain data (1-D array)
            fs       -> Sampling frequency
            n        -> Order of the filter
            band     -> Frequency band [fmin, fmax]
            doplots  -> Do plots for checks (0: no, 1: yes)
            
    OUTPUT:
            h_filt   -> Filtered strain data
    """

    # ------> Compute Butterworth filter's coefficients
    
    Nyq = fs/2   # Nyquist frequency
    (b, a) = signal.butter(n, band/Nyq, btype='bandpass')
    # b: input coefficients
    # a: output coefficients
    
    print("****** Butterworth filter's coefficients  ******")
    print("")
    print("Input:", b)
    print("Output:", a)
    print("")
    
    # ------> Check: Frequency response of the filter
    
    if doplots:
        
        print("****** Check: Filter frequency response ******")
        ffreq, fresponse = freqz(b, a, fs=Nyq)

        plt.figure(1, figsize=(8,5))
        plt.plot(ffreq, np.abs(fresponse)) # Response in gain
        #plt.plot(ffreq, 20 * np.log10(abs(fresponse))) # Response in decibels
        plt.xscale('log')
        plt.title("Filter frequency response", fontsize=18)
        plt.ylabel("Amplitude", fontsize=13)
        plt.xlabel("Frequency (Hz)", fontsize=13)
        #plt.grid(True)
        plt.show()
        plt.figure(1).clear()
        gc.collect()
        print("")
    
    # ------> Apply Butterworth filter to the signal
        
    h_filt = signal.filtfilt(b, a, h)
    
    return h_filt

# ---------------------------------------------------------------
# ------> Band pass filter with PyCBC library
# ---------------------------------------------------------------

def filt_bandpass_PyCBC(s, n, band, ts, w_seg, doplots):
        
    """
    Function that applies a band pass filter to strain data,
    using the standard LIGO PyCBC library.
    
    INPUT:
            s        -> Strain data (1-D array)
            n        -> Order of the filter
            band     -> Frequency band [fmin, fmax]
            ts       -> Sampling time
            w_seg    -> Duration of each Welch segments (in seconds)
            doplots  -> Do plots for checks (0: no, 1: yes)
            
    OUTPUT:
            s_filt_t -> Filtered strain data
            psd_filt  -> PSD of the filtered strain data
            fpsd_filt -> PSD_filt's frequencies
    """

    # Convert strain numpy array to timeseries
    s = pycbc.types.timeseries.TimeSeries(s, ts)

    # Recover PSD of whitened strain data
    seg_len    = int(w_seg/ts)
    seg_stride = int(seg_len/2)
    psd_white = pycbc.psd.welch(s, seg_len=seg_len, seg_stride=seg_stride)
    fpsd_white = psd_white.sample_frequencies
    
    # Remove low frequencies
    s_filt = pycbc.filter.highpass(s, band[0], n)
    #plt.plot(s_filt.sample_times, s_filt, label='Highpassed')

    # Remove high frequencies
    s_filt = pycbc.filter.lowpass(s_filt, band[1], n)
    #plt.plot(s_filt.sample_times, s_filt, label='Highpassed + Lowpassed')

    # Estimate Bandpass PSD
    seg_len    = int(w_seg/ts)
    seg_stride = int(seg_len/2)
    
    psd_filt = pycbc.psd.welch(s_filt, seg_len=seg_len, seg_stride=seg_stride)
    fpsd_filt = psd_filt.sample_frequencies
        
    if doplots:
        
        #mpl.rcParams['agg.path.chunksize'] = 10000
        print("****** Check: Plot ASDs ******")
        
        plt.figure(1, figsize=(7.75,5))
        plt.plot(fpsd_white, np.sqrt(psd_white), label='ASD of whitened data')
        plt.plot(fpsd_filt, np.sqrt(psd_filt), label='ASD of filtered data')
        #plt.xlim(left=9)
        plt.xscale('log')
        plt.yscale('log')
        plt.title("PSD", fontsize=18)
        plt.ylabel("Power Spectral Density", fontsize=13)
        plt.xlabel("Frequency (Hz)", fontsize=13)
        plt.legend()
        plt.show()
        plt.figure(1).clear()
        gc.collect()
        print("")
    
    else:
        pass

    return s_filt, psd_filt, fpsd_filt

# ---------------------------------------------------------------
# ------> Wavelet time-frequency representation
# ---------------------------------------------------------------

def WaveletTF_transform(h, fs, fstart, fstop, delta_f, width, doplots):

    """
    This function computes the time-frequency representation
    based on a Wavelet transform of a signal (time series)

    INPUT:
            h        -> Signal
            fs       -> Sampling frequency
            fstart   -> Filter's initial frequency
            fstop    -> Filter's final frequency
            delta_f  -> Frequency bin width
            width    -> Width of the wavelet (in cycles)
            doplots  -> Do plots for checks (0: no, 1: yes)
            
    OUTPUT:
            timeVec  -> Vector of times
            freqVec  -> Vector of frequencies
            WL       -> Wavelet coefficients
    
    """
        
    # ------> Time vector and time sampling
        
    Nsamples = len(h)
    timeVec = np.arange(0, Nsamples)/fs
    ts      = 1/fs    

    # ------> Frequency vector
        
    Nfreq = round((fstop - fstart) / delta_f) + 1
    freqVec = np.linspace(fstart, fstop, Nfreq).reshape((-1, 1))
    
    # ------> Initialize Wavelet Transform Matrix
        
    WL = np.zeros((Nfreq, Nsamples))

    #print("Wavelet Transform Matrix shape:", WL.shape)
        
    # ------> Compute the time-frequency representation

    signal = h
    #signal = detrend(h, axis=-1, type='linear')

    for ifre in range(Nfreq):
                
        # Compute the Morlet wavelet
        Morlet = Morlet_wavelet(freqVec[ifre], ts, width, doplots);
        
        # Apply the Morlet wavelet transform
        WLcomplex = np.convolve(signal, Morlet, mode='full')
        
        # Get indexes
        li = int(np.ceil(len(Morlet) / 2))
        ls = li + Nsamples
        #ls = len(WLcomplex) - int(np.flor(len(Morlet) / 2))

        # Complex coefficients
        WLcomplex = WLcomplex[li:ls]
        
        if doplots:
                
            print("Frecuency =", freqVec[ifre])
            print("++++++++++++++++++++++++++++++++++++++++")
            print("")
            
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(np.real(WLcomplex), 'r')
            plt.plot(np.imag(WLcomplex), 'b')
            plt.legend(['real', 'imag'])
            plt.box(True)
            plt.title('Frequency: ' + str(freqVec[ifre]) + ' Hz')
                
            plt.subplot(3, 1, 2)
            plt.plot(np.abs(WLcomplex))
            plt.legend(['magnitude'])
            plt.box(True)

            plt.subplot(3, 1, 3)
            plt.plot(np.angle(WLcomplex))
            plt.legend(['phase'])
            plt.box(True)
                
            plt.tight_layout()
            plt.pause(0.1)  # Pause for a short time to show the plot
            #plt.savefig('wavelet_decomposition.png')

        # Compute the magnitude
        WLmag = 2*(np.abs(WLcomplex)**2)/fs
        
        # Save wavelet decomposition magnitude
        WL[ifre, :] = WLmag
        
        WL = np.squeeze(WL)
        
        if doplots:
        
            plt.figure()
            X, Y = np.meshgrid(timeVec, freqVec)
    
            plt.pcolormesh(X, Y, WL[:, :], shading='gouraud')
    
            plt.axis([np.min(timeVec), np.max(timeVec), np.min(freqVec), np.max(freqVec)])
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Time-Frequency representation')
            plt.box(True)
            plt.colorbar()

            plt.show()
    
    return timeVec, freqVec, WL

# ---------------------------------------------------------------
# ------> Morlet Wavelet time-frequency representation
# ---------------------------------------------------------------

def Morlet_wavelet(fi, ts, morlet_w, doplots):
    
    """
    Function to compute the Complex Morlet Wavelet for frequency "fi"
    and time "t". This will be normalized so the total energy = 1.
    
    INPUT:
            fi        -> Frequency
            ts        -> Sampling time
            morlet_w  -> Wavelet's width (width>= 5 is suggested) 
            doplots   -> Do plots for checks (0: no, 1: yes)
            
    OUTPUT:
            Morlet    -> Morlet wavelet
    """ 

    # Frequency standard deviation
    sf = fi / morlet_w
    
    # Time standard deviation 
    st = 1/(2*np.pi*sf)   

    # Wavelet's amplitude
    A  = 1/np.sqrt(st*np.sqrt(np.pi))

    # Time array
    t = np.arange(-3.5 * st, 3.5 * st + ts, ts)

    # Compute Morlet wavelet
    Morlet =  A*np.exp(-t**2/(2*st**2))*np.exp(1j*2*np.pi*fi*t)

    if doplots:
        
        plt.figure()
        
        plt.subplot(3, 1, 1)
        plt.plot(t, np.real(Morlet), 'r', linewidth=2)
        plt.plot(t, np.imag(Morlet), 'b', linewidth=2)
        plt.ylabel('Morlet')
        plt.legend(['Real', 'Imag'])
        plt.box(True)
        plt.title('Frequency = ' + str(fi) + 'Hz')
        
        plt.subplot(3, 1, 2)
        plt.plot(t, np.abs(Morlet), '.-r')
        # plt.axis([-4, 4, 0, 6])
        plt.xlabel('Time (s)')
        plt.ylabel('Magnitude')
        
        plt.subplot(3, 1, 3)
        plt.plot(t, np.angle(Morlet), '.-b')
        # plt.axis([-4, 4, -4, 4])
        plt.xlabel('Time (s)')
        plt.ylabel('Angle')
        
        plt.tight_layout()
        plt.show()

    return Morlet

# ---------------------------------------------------------------
# ------> Window sample log files reader
# ---------------------------------------------------------------

def load_logdata(folder_path, populations):
  
    """
    Function to load populations data from final log files.
    
    INPUT:
            folder_path     -> Folder path
            populations     -> Dictionary, to save log data
                               for each class (key)
            
    OUTPUT:
            populations     -> Updated dictionary
            
            Num_injections  -> No. of loaded injections
    """
        
    os.chdir(folder_path)
    
    # Initialize No. injections count
    Num_injections = 0
    
    # Loop: run along subfolders of waveforms' class
    for subfolder in glob.glob("wfclass_*"):
        
        print("")
        print("=======> SCANNING", subfolder, "SUBFOLDER")
        print("")
        
        class_label = subfolder[-1]
        
        print("READING LOG DATA FILE")
        
        # ------> Initialize lists for populations

        t_inj = []
        jitter = []
        wf_SNR = []
        Slope = []
        f_ini = []
        f_end = []
        wf_duration = []
        log_data = []
        
        log_location = folder_path + "/" + subfolder + "/log.dat"
        with open(log_location) as csv_file:
                
            print("Log file location:", log_location)
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
        
            for row in csv_reader:
                if line_count == 0:
                    #print(f'Data columns: {", ".join(row)}')
                    line_count += 1
                else:
                    t_inj.append(row[0])
                    jitter.append(row[1])
                    wf_SNR.append(row[2])
                    Slope.append(row[3])
                    f_ini.append(row[4])
                    f_end.append(row[5])
                    wf_duration.append(row[6])
                    
                    line_count += 1
                
        print("")
        print(f'Processed {line_count} lines')
        print("")
        
        log_data.append([t_inj, jitter, wf_SNR, Slope, f_ini, f_end, wf_duration])
        
        populations["class " + class_label] = log_data[0]
        
        Num_injections += line_count
        
    print("")
    return populations, Num_injections

# ---------------------------------------------------------------
# ------> HFF slope checker of a waveform
# ---------------------------------------------------------------

def check_class_HFFslope(HFFslope):
  
    """
    Simple function to check the class of a waveform
    according to its HFF slope
    
    INPUT:
            slope  -> HFF slope of the waveform
            
    OUPUT:
            wf_type
    """

    if 950 <= HFFslope < 1450:
        wf_type = 3
        print("Waveform is in class 3")
        print("+++++++++++++++++++++++++++")
    
    elif 1450 <= HFFslope < 1620:
        wf_type = 2
        print("Waveform is in class 2")
        print("+++++++++++++++++++++++++++")
    
    elif 1620 <= HFFslope <= 4990:
        wf_type = 1
        print("Waveform is in class 1")
        print("+++++++++++++++++++++++++++")
    
    else:
        wf_type = float("NaN")
        print("HFF slope is out of range")
        print("+++++++++++++++++++++++++++")
        
    return wf_type

# ---------------------------------------------------------------
# ------> High pass filter for numerical multidim waveforms
# ---------------------------------------------------------------

def filt_highpass_NumWF(h, fs, n, fc, doplots):
  
    """
    Function that applies a high pass filter to numerical
    multidimensional waveforms to remove the SASI contribution
    
    INPUT:
            h       -> Strain data (1-D array)
            fs      -> Sampling frequency 
            n       -> Order of the filter
            fc      -> Critical frequency
            doplot  -> Do plots for checks (0: no, 1: yes)
            
    OUPUT:
            h_filt -> Filtered strain data
    """
        
    # ------> Compute Butterworth filter's coefficients
    
    Nyq = fs/2   # Nyquist frequency
    (b, a) = signal.butter(n, fc/Nyq, btype='highpass')
    # b: input coefficients
    # a: output coefficients
    
    print("****** Butterworth filter's coefficients  ******")
    print("")
    print("Input:", b)
    print("Output:", a)
    print("")
    
    # ------> Check: Frequency response of the filter
    
    if doplots:
        
        print("****** Check: Filter frequency response ******")
        ffreq, fresponse = freqz(b, a, fs=Nyq)

        plt.figure(1, figsize=(7.5,5))
        plt.plot(ffreq, np.abs(fresponse)) # Response in gain
        #plt.plot(ffreq, 20 * np.log10(abs(fresponse))) # Response in decibels
        plt.xscale('log')
        plt.title("Filter frequency response", fontsize=18)
        plt.ylabel("Amplitude", fontsize=13)
        plt.xlabel("Frequency (Hz)", fontsize=13)
        #plt.grid(True)
        plt.show()
        plt.figure(1).clear()
        gc.collect()
        print("")
    
    # ------> Apply Butterworth filter to the signal
        
    h_filt = signal.filtfilt(b, a, h)
    
    return h_filt
        

# ---------------------------------------------------------------
# ------> Function to compute waveform strain
# ---------------------------------------------------------------

def compute_strain(t,hplus,hcross,theta,phi):

    # t              -> time array
    # hplus, hcross  -> polarization strain arrays
    # theta, phi     -> observation angles
    
    # Compute detector pattern functions
    Fp = 0.5 * (1. + np.cos(theta)**2) * np.cos(2*phi)
    Fc = np.cos(theta) * np.sin(2*phi)
        
    # Compute strain function
    ho = Fp*hp + Fc*hc
    
    # h -> strain function h(t)
    return ho

# ---------------------------------------------------------------