# Summary

Dataset generator of strain (time series) samples containing phenomenological gravitational waves from core-collapse supernovae added to noise from LIGO-Virgo interferometric detectors. Jupyter notebooks developed in Python and written by Manuel D. Morales (e-mail: <manueld.morales@academicos.udg.mx>).


# Astrophysical details

<b><ins>Interferometric noise</ins></b>

For this implementation, we used real noise from LIGO (L1, H1) and Virgo (V1) interferometric detectors from O3b run. This is a non-Gaussian and non-stationary noise stored in strain time series of 4,096s of duration, sampled at 16384Hz and 4096Hz. For our implementation, we choose the second sampling option. Data available on the [Gravitational Wave Open Science Center](https://gwosc.org/).

<b><ins>Phenomenological waveforms</ins></b>

We draw on CCSN phenomenological waveforms. These were generated by the [SignalGenerator_Phen](https://github.com/CesarTiznado/SignalGenerator_Phen) code, following the model first proposed by [Astone et al Phys. Rev. D 98, 122002 (2018)](https://doi.org/10.1103/PhysRevD.98.122002). This simplified stochastic non-physical model mimics one of the features common in GW signals obtained from all multi-dimensional general relativistic CCSN simulations, namely the high-frequency feature (HFF) -commonly called as "g-mode". In the time-frequency representation (such as that provided by spectrograms and scalograms), the HFF has an increasing monotonically frequency profile in time, which, at first order, can be considered linear. We draw on three sets of phenomenological waveforms:

- 200 waveforms with 1,620 =< HFF slope =< 4,990 (class 1)
- 200 waveforms with 1,450 =< HFF slope < 1,620 (class 2)
- 200 waveforms with 950 =< HFF slope < 1,450 (class 3)

# Implementation structure

```
datagen-sngw-phen
|___ Codes
     |___ Noise_Explorer.ipynb
     |___ Prepare_Waveforms.ipynb
     |___ Process_PhenWaveforms.ipynb
     |___ Toolbox.py
     |___ WaveformPhen_Explorer.ipynb
|___ Datasets
|___ Waveforms_mod
     |___ Phen
|___ Waveforms_orig
     |___ Phen
|___ LICENCE
|___ README.md
```

Run the Jupyter notebooks in the following order:

`Noise_Explorer.ipynb` for downloading interferometric noise data from LIGO and Virgo detectors (segments of 4,096s), exploring time series and power spectral densities, and locally saving noise segments. Open data was obtained from the website of the GWOSC.

`WaveformPhen_Explorer.ipynb` for analyzing three representative waveforms; exploring their morphologies in the time domain and the time-frequency domain. Spectrograms (applying short-time Fourier transform) and scalograms (applying Morlet wavelet transform) are plotted.

`Prepare_Waveforms.ipynb` for resampling and rescaling all phenomenological waveforms, resulting in dimensionless strain time series with a sampling frequency of 4,094Hz. Modified waveforms are saved as dictionaries, separated by class.

`Process_PhenWaveforms.ipynb` for applying injections, data conditioning (whitened, band-pass filtering), and generating a dataset of window strain samples. Each one of these windows has the same length Twin<4,096s, contains noise plus a phenomenological waveform, and is saved in a .txt file. In addition, a log.dat file is saved which contains information about injected signals in all windows.

`Populations.ipynb` for exploring the distribution of SNR values, HFF slopes, waveform durations, and frequency ranges f1-f0 of the HFF. This information is loaded from log.data file. SNR values are computed from the strain samples containing noise, and other quantities from waveforms before being injected.

In addition, the file `Toolbox.py` is included, which contains specific functions for the notebooks.

# Important instructions

1. All scripts were run locally, then you will need to edit path locations in cells for read files. The set of 600 phenomenological waveforms used in this work is available in the public folder [Waveforms_orig](https://drive.google.com/drive/folders/1GuOWzGEHlAedqWZcCAShcAanpDnC1bIy?usp=sharing). Download this folder and locate it in your local machine as shown in the tree detailed in the previous section.

2. In the input of `Prepare_Waveforms.ipynb` script, you can change the rescaling factor (Resc_factor) to change the SNR values of injected signals. Depending on this rescaling factor (and the noise realization), the distribution of SNR values can change. To explore this distribution, run the `Populations.ipynb` script.

3. In the `Process_PhenWaveforms.ipynb` script, a random combination of waveforms is selected to be injected in a noise strain data segment. Each injection is performed in a location inj_time +/- jitter (in seconds), in which jitter is a random value in the range [inj_time - jitter_lim, inj_time + jitter_lim]. Jitter_lim is an input parameter that you can change in subsection 2.1 of the aforementioned script (ensure that jim_lim << dt_inj).

4. The time window duration of window strain samples is Twin = wf_max + alpha, where wf_max is the duration of the longest injected waveform, and alpha is an input parameter. Then, to change window duration, change alpha input parameter in subsection 4.1 of `Process_PhenWaveforms.ipynb` script.

5. Before running `Process_PhenWaveforms.ipynb` to perform injections (and to generate window samples) covering a whole noise segment of 4,096s, it is highly recommendable to apply it only on a reduced segment of a few tens of seconds. To perform this check, set reduce_segment=1 and reduced_time_n equal to the duration (in seconds) of the reduced segment, in subsection 2.1 of the script.
   
6. For the check mentioned in instruction 5, it could be also useful to set doplot_spectrogram=1 in the first cell of section 3.4.1, and set_doplots=0 in the first cell of section 4.1, to visualize window samples in time and time-frequency domains. For the complete process of generating window strain samples, this is not recommendable, because outputting plots considerably slows down the execution of the script.
