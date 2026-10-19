# DyadDAQP.jl

DyadDAQP is the package DAQPBase of [DAQP](https://github.com/darnstrom/daqp), renamed and
registered in DyadRegistry. It is built from the branch `dyad` of
[JuliaComputing/daqp](https://github.com/JuliaComputing/daqp), which contains changes that are not
yet part of a DAQP release, and it loads `libdaqp` from `DyadDAQP_jll`, which is built from the same
branch. The API is that of DAQPBase.

A Julia wrapper for the Quadratic Programming solver [DAQP](https://github.com/darnstrom/daqp). Is used by the Julia package [DAQP.jl](https://github.com/darnstrom/DAQP.jl
