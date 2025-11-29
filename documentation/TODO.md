# TODO LIST FOR PYCAUSET


Goals:
- [ ] Calculate eq. (3.5) in Johnston: $K = \Phi (I - b\Phi)^{-1}$ 
- [ ] Calculate a 100GB $K$
- [ ] Numpy compatibility

## Wanted features
- [x] Saving/loading causal matrices to/from binary files.
- [ ]  Implement matrix operations: addition, subtraction, multiplication.
- [ ]  Implement inverting
- [ ] Identity matrix: pycauset.I - it should detect size automatically.
- [ ] Add the option to specify where to save/load the binary files - especially from external drives
- [x] Make binary files have .pycauset extension
- [ ] Allow users to specify lower size of elements which get turned into and stored as binary files
- [ ] Matrix printing
- [ ] Figure out how to correctly randoply populate C

