# TODO LIST FOR PYCAUSET


Goals:
- [x] Calculate eq. (3.5) in Johnston: $K = \Phi (I - b\Phi)^{-1}$ 
- [ ] Calculate a 100GB $K$
- [x] Numpy compatibility
- [ ] Study the base changes and figure out what to do with the eigenvectors
- [x] Host the documentation (github site?)

## Wanted features
- [x] Figure out how to correctly randomly fill C
- [ ] Visualizations of causets
- [ ] gpu parallelization
- [x] Saving/loading causal matrices to/from binary files.
- [x] Implement matrix operations: addition, subtraction, multiplication.
- [x] Implement inverting
- [x] Identity matrix: pycauset.I - it should detect size automatically.
- [x] Add the option to specify where to save/load the binary files - especially from external drives
- [x] Make binary files have .pycauset extension
- [x] Allow users to specify lower size of elements which get turned into and stored as binary files (deprecated)
- [x] Matrix printing
- [x] printable "info" about matrix. I.e. print info from header
- [x] Fix bug so that binaries are actually deleted on program finish
- [x] add dtype specification on matrix creation
- [x] vectors
- [x] complex numbers support
- [x]  introduce RAM-backed mapping for small temporary objects
- [ ] Create Pauli-Jordan function $i\Delta$ 
- [ ] User-configurable spacetimes, geometries and sprinklings
- [ ] DISABLE AUTO-versioning being pushed to pypi!!!!!

### Qs:
- [ ] When a CausalSet instances is created, how does one ensure/know that the CausalMAtrix has been saved? Make this intuitive!
- [ ] what do we do with pc.CausalMatrix(N, populate=True) now that "populate" uses and outdated version of population? 

## Documentation needed
- [x] pycauset.save (deprecated)
- [x] pycauset.TriangularMatrix
- [x] pycauset.keep_temp_files
- [x] pycauset.Vector
- [x] pycauset.set_memory_threshold
- [ ] `$PYCAUSET_STORAGE_DIR`
