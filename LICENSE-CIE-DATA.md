# CIE spectral datasets

The official CSV data, metadata, and the exact numeric JavaScript reformat in
`dist/plt-cie.js` are separately licensed under Creative Commons
Attribution-ShareAlike 4.0 International (CC BY-SA 4.0):
https://creativecommons.org/licenses/by-sa/4.0/

- CIE 2019, CIE 1931 colour-matching functions, 2 degree observer,
  International Commission on Illumination (CIE), Vienna, AT.
  DOI: https://doi.org/10.25039/CIE.DS.xvudnb9b
- CIE 2019, CIE standard illuminant D65,
  International Commission on Illumination (CIE), Vienna, AT.
  DOI: https://doi.org/10.25039/CIE.DS.hjfjmt59

Changes: CSV numeric data have been reformatted as JavaScript arrays; numeric
values are unchanged. Both datasets are used with linear interpolation and
zero extrapolation. Official metadata/checksums accompany the original CSVs in
`dist/spectral-data/` and `research/spectral-data/`. No endorsement is implied.
The PLT code's separate licensing remains stated in LICENSE-WAVE-TRACING.md.
