# Map and Vector Field Transformations

The following methods calculate invariant foliations and invariant manifolds of maps and vectopr fields.

Let us consider the following four-dimensional map
```julia
function trivialMAP(z)
    om1 = 0.12
    om2 = 1
    dm1 = 0.002
    dm2 = 0.003
    return [(exp(dm1)*z[4]*(z[1]^3*z[2] + z[4]^2 + z[4]^4) + 2*z[1]*cos(om1) - 2*z[2]*sin(om1))/(2.0*exp(dm1)),
            -0.5*z[2]^3 - z[1]^2*z[2]^2*z[4] - (3*z[2]*z[3]*z[4])/4. + (z[2]*cos(om1) + z[1]*sin(om1))/exp(dm1),
            (-3*z[1]*z[2]^3*z[3])/4.0 + z[3]^5 - (3*z[1]^3*z[2]*z[4])/4.0 + (z[3]*cos(om2) - z[4]*sin(om2))/exp(dm2),
            (z[2]*z[3]^4)/4.0 + (z[2]*z[3]^3*z[4])/2.0 - z[4]^5 + (z[4]*cos(om2) + z[3]*sin(om2))/exp(dm2)]
end
```
To calculate the invariant foliation corresponding to a two-simensional invariant subspace we use
```julia
using FoliationsManifoldsAutoencoders

MF = DensePolyManifold(4, 4, 5)
XF = fromFunction(MF, trivial)

MWt, XWt, MS, XS = iFoliationMAP(MF, XF, [3, 4], [])
```
To calculate the invariant manifold corresponding to the same subspace we use
```julia
MWt, XWt, MR, XR = iManifoldMAP(MF, XF, [3, 4], [])
```
One can also calculate the frequency and damping curves using
```julia
Dr = 0.0001
r = range(0,1,step=Dr)
opscal = ones(1,4)/4
frequency, damping, amplitude = MAPManifoldFrequencyDamping(MWt, XWt, MR, XR, r, 1.0; output = opscal)
```

The relevant functions are the following:

```@docs
iFoliationMAP
```

```@docs
iFoliationVF
```

```@docs
iManifoldMAP
```

```@docs
iManifoldVF
```
