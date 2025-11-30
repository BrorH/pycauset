# TODO LIST FOR PYCAUSET


Goals:
- [x] Calculate eq. (3.5) in Johnston: $K = \Phi (I - b\Phi)^{-1}$ 
- [ ] Calculate a 100GB $K$
- [ ] Numpy compatibility

## Wanted features
- [x] Saving/loading causal matrices to/from binary files.
- [x] Implement matrix operations: addition, subtraction, multiplication.
- [x] Implement inverting
- [x] Identity matrix: pycauset.I - it should detect size automatically.
- [x] Add the option to specify where to save/load the binary files - especially from external drives
- [x] Make binary files have .pycauset extension
- [x] Allow users to specify lower size of elements which get turned into and stored as binary files (deprecated)
- [x] Matrix printing
- [ ] Figure out how to correctly randomly fill C
- [ ] Visualizations of matrices
- [x] printable "info" about matrix. I.e. print info from header
- [x] Fix bug so that binaries are actually deleted on program finish
- [x] add dtype specification on matrix creation
- [ ] vectors
- [ ] complex numbers support


## Documentation needed
- [x] pycauset.save (deprecated)
- [ ] pycauset.TriangularMatrix
- [ ] pycauset.keep_temp_files
- [ ] `$PYCAUSET_STORAGE_DIR`
