# A complex ten-dimensional system

The data is created by
```
cd("data")
include("state_space_generate.jl")
include("pca_dft_generate.jl")
```
The foliations can be calculated for the full state-space data using
```
cd("..")
include("fullstate-amp-1.jl")
include("fullstate-amp-1.jl")
include("fullstate-amp-1.jl")
include("fullstate-amp-1.jl")
```
For the PCA reconsructed state-space
```
include("pcastate-amp-1.jl")
include("pcastate-amp-2.jl")
include("pcastate-amp-3.jl")
include("pcastate-amp-4.jl")
```
For the perfect reproducing filter bank data (DFT)
```
include("dft_state-1.jl")
include("dft_state-2.jl")
include("dft_state-3.jl")
include("dft_state-4.jl")
```
Koopman eigenfunctions are calculated by 
```
include("koopman-amp-1.jl")
include("koopman-amp-2.jl")
include("koopman-amp-3.jl")
include("koopman-amp-4.jl")
```
Generate data with initial conditions from the invariant manifold
```
include("aenc-null-generate.jl")
```
Fit an autoencoder to the data on the invariant manifold
```
include("aenc-tautology.jl")
```
Try to fit an autoencoder to properly generated data (and see it fail)
```
include("aenc-densedata.jl")
```
Finally, plot the results
```
include("plotresults.jl")
```
