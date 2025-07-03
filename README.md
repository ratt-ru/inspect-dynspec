# inspect-dynspec
A Radio Astronomy Software post-processing tool for [RIMS](https://github.com/saopicc/RIMS) _Dynamic Spectra_ output. The tool will produce a noise estimate and smooth the target dynamic spectra according to the provided RIMS output and specified parameters.
In essence, it serves as a companion tool to RIMS, where it will improve and neatly plot (`png`) the dynamic spectra (`fits`) from RIMS.

### Installation

#### Steps:
```
git clone https://github.com/ratt-ru/inspect-dynspec.git
pip install ./inspect-dynspec
```

### Usage:
View help:
```
inspect-dynspec --help
```
Run job:
```
inspect-dynspec <path-to-RIMS-output> --kernel 20,5 --kernel 10,10 --output <path-to-desired-output-folder> --stokes IV --std-scale 2.5
```
Notes:
- You may specify multiple smoothing kernels as shown above.
- You may specify which Stokes dynamic spectra you want processed and plotted.
- The specified output directory will be created if it is not.
- The input path need not point exactly to the output of RIMS folder, as the tool will traverse down until it finds a RIMS-like output directory (to make sure it gets the right one though, you should probably be exact).

### Output:
Depending on your parameters, the output folder will contain an assortment of `png` plots.
Suffix identification:
- `*_a_e_denoise.png` : A beam/Jy fully denoised (analytically and excess) and smoothed target dynamic spectra - usually the product you want.
- `*_W.png` : A plot of the Weights produced by RIMS (used in analytical denoising).
- `*_W2.png` : A plot of the Weights^2 produced by RIMS (used in analytical denoising).
- `_flagged_regions.png` : A mask plot of scan breaks and flagged frequencies.
- `_a_denoise.png` : Analytically denoised and smoothed target dynamic spectra (beam/Jy).
- `_rawdata.png` : The smoothed raw target dynamic spectra (beam/Jy).
- `_SNR.png` : A fully denoised and smoothed target dynamic spectra with the smoothed calculated `standard deviation` divided out to give an SNR plot.
- `_var_a.png` : The analytical variance calculated from the W and W2 weights. It is applied to the `rawdata` to produce the `analytically denoised` product.
- `_var_e.png` : The excess variance calculated from the RIMS off target dynamic spectra. It is applied to the `analytically denoised` product to produce the fully denoised product.

### Licensing:
MIT License

Copyright (c) 2025, Talon Myburgh, Cyril Tasse, Oleg Smirnov, Landman Bester, Rhodes University Centre for Radio Astronomy Techniques & Technologies (RATT), Observatoire de Paris