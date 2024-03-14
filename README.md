# Summary

Dataset generator of strain (time series) samples containing phenomenological gravitational waves from core-collapse supernovae added to noise from LIGO-Virgo interferometric detectors. Jupyter notebooks developed in Python and written by Manuel D. Morales (e-mail: <manueld.morales@academicos.udg.mx>).


# Technical details

<b><ins>Interferometric noise</ins></b>

For this implementation, we used real noise from LIGO (L1, H1) and Virgo (V1) interferometric detectors from O3b run. This is a non-Gaussian and non-stationary noise stored in strain time series of 4,096s of duration, sampled at 16384 Hz and 4096 Hz. For our implementation, we choose the second sampling option. Data available on the [Gravitational Wave Open Science Center](https://gwosc.org/).

<b><ins>Phenomenological waveforms</ins></b>

We draw on CCSN phenomenological waveforms. These were generated by the [SignalGenerator_Phen](https://github.com/CesarTiznado/SignalGenerator_Phen) code, following the model first proposed by [Astone et al Phys. Rev. D 98, 122002 (2018)](https://doi.org/10.1103/PhysRevD.98.122002). This is a simplified stochastic non-physical model that mimics one of the features that is common in GW signals obtained from all multi-dimensional general relativistic CCSN simulations, namely the high-frequency feature (HFF). In the time-frequency representation (such as that provided by spectrograms and scalograms), the HFF has an increasing monotonically frequency profile in time, which, at first order, can be considered linear. We draw on three sets of phenomenological waveforms:

- <b>Class 1</b>: 200 waveforms with $1,620 \lt \text{HFF slope} \lt 4,990$
- <b>Class 2</b>: 200 waveforms with $1,450 \lt \text{HFF slope} \lt 1,620$
- <b>Class 3</b>: 200 waveforms with $950 \lt \text{HFF slope} \lt 1,450$

# Implementation structure

Run the codes in the following order:

`Noise_Explorer.ipynb`</br>
Download noise data, explore strain time series and power spectral densities, and locally save noise segments.

`WaveformPhen_Explorer.ipynb`</br>
Explore three representative waveforms, exploring their HFF in time and time-frequency domains. 

`Prepare_Waveforms.ipynb`</br>
Resample waveforms to the same sampling frequency of noise, then rescale waveforms to have adimensional strain magnitudes.
