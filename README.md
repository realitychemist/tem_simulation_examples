# (S)TEM Simulation Examples
----

__This project is now archived; the abTEM examples do not work on current versions, due to major breaking changes in how abTEM works.__

This repository contains several examples demonstrating how to perform a few different types of (S)TEM simulations.  It covers two programs:

1. Dr Probe -- A GUI-focused program that uses the multislice algorithm in its simulations.  Dr Probe is straightforward to use and most tasks can be performed using the GUI, although a command prompt-based approach is also available.  Dr Probe is available  [on this website](https://er-c.org/barthel/drprobe/).  It comes with several supporting software packages for building and editing atomic structure models (which are command-line based).
2. abTEM -- A Python library which is able to perform simulations using either multislice or PRISM.  abTEM is highly flexible as it integrates directly into Python code, and uses the Python library `ASE` (atomic simulation environment) to build and edit atomic structure models.
