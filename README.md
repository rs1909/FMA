# A Julia package to calculate invariant foliations, invariant manifolds and autoencoders

There are two examples for this package, a [10 dimensional synthetic system](examples/synthetic_ten_dimensional/README.md) and 
a [jointed beam](examples/jointed_beam/README.md)

Before running the examples you need to make sure that the package in in the load path of Julia. This can be done by
```
push!(LOAD_PATH,pwd())
```
in the root directory of the package.

Note that the examples are computationally intensive. It is advisable to run various commands in parallel, to speed up the process.
