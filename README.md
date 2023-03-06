# Simple Beam

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Introduction
A basic beam bending analysis package, intended to do simple beam bending moment & shear force analysis. The method used is McCauley's method, and the implementation is based on SymPy's beam analysis module, just more nicely wrapped for everyday usage.

The use of the term "Simple" does not mean that only simply-supported beams will be included in this package. McCauley's method can handle fixed ended (and even multi-span)
 beams. However, beams with axial loads, beams in 3-dimensions and frames, trusses etc. will not be included.

This is primarily intended to be a proof of concept package for me - at this point it is not a robust means for doing your engineering analysis. This may change as the package develops.

# Disclaimer
While all efforts have been made to ensure that the appropriate engineering theories etc. have been implemented correctly, it is the user's responsibilty to ensure that all output is correct. In particular, users should be familiar with basic structural mechanics and standard engineering practices. For example, doing independent checks of tools you take from unknown authors on the internet.

# Installation

Use your preferred virtual environment solution and then simply pip install.

```pip install simplebeam```

# Basic Usage
*To be filled out*

# Documentation
You're reading it. Additionally, check the tests folder to see additional uses. I may add documentation at some future point in time. 

# Future Development
The following future developments *may* be done:

- [ ] Implementation of helper methods for different load types.
- [ ] Multiple load cases & load combinations
- [ ] Implementation of beams with pins & varying properties.

# Contributing
Feel free to contribute through a pull request to provide bug fixes, new features or documentation.

Note that the intent of this program is that it will remain focussed on simple 2D beam bending only.