# Simple Beam

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A basic beam bending analysis package, intended to do very simply beam bending moment & shear force analysis. The method used is McCauley's method, and the implementation is based on SymPy's beam analysis module, just more nicely wrapped for everyday usage.

The use of the term "Simple" does not mean that only simply-supported beams will be included in this package. McCauley's method can handle fixed ended (and even multi-span)
 beams. However, beams with axial loads, beams in 3-dimensions and frames, trusses etc. will not be included.

This is primarily intended to be a proof of concept package for me - at this point it is not a robust means for doing your engineering analysis. This may change as the package develops.

Users should be familiar with basic structural mechanics and standard engineering practices such as doing independent checks of tools you take from unknown authors on the internet.