# An example for calculating an invariant foliation

The following code calculates an invariant foliation of a 4-dimensional map and compares it to the analytic calculation by `iManifoldMAP`.

First we bring the required packages into scope.
```julia
using FoliationsManifoldsAutoencoders
using Manifolds
using Plots
```

Then we define a discrete-time map, which we use to calculate invariant foliations and invariant manifolds for.
```julia
function trivial(z)
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

We define a function that creates the required data. This function iterates `trivial` with random initial conditions.
```julia
function generate(amp = 1.0)
    ndim = 4
    nruns = 16000
    npoints = 1
    xs = zeros(ndim, nruns*npoints)
    ys = zeros(ndim, nruns*npoints)
    aa = rand(ndim,nruns) .- ones(ndim,nruns)/2
    ics =  aa ./ sqrt.(sum(aa.^2,dims=1)) .* (2*rand(1,nruns) .- 1)
    
    for j=1:nruns
        u0 = ics[:,j] * amp
        for k=1:npoints
            u1 = trivial(u0)
            xs[:,k+(j-1)*npoints] .= u0
            ys[:,k+(j-1)*npoints] .= u1
            u0 .= u1
        end
    end
    return xs, ys
end
```

We convert map `trivial` into polynomial form.
```julia
MF = DensePolyManifold(4, 4, 5)
XF = fromFunction(MF, trivial)
```

We calculate the invariant manifold corresponding to the 3rd and 4th eigenvalues of the Jacobian of `trivial`, which form a complex conjugate pair.
```julia
MWt, XWt, MS, XS = iManifoldMAP(MF, XF, [3, 4], [])
```

We calculate the corrected instantaneous frequencies and damping ratios for the invariant manifold, we have just calculated.
```julia
amp_max = 0.15
opscal = ones(1,4)/4
That, Rhat_r = MAPFrequencyDamping(MWt, XWt, MS, XS, amp_max, output = opscal)
r = range(0, domain(That).right, length=1000)
omega = abs.(That.(r))
zeta = -log.(abs.(Rhat_r.(r))) ./ abs.(That.(r))
```

We generate data.
```
# create data
dataIN, dataOUT = generate()
```

We identify the invariant foliation, locally invariant foliation, extract the invariant manifold and calculate instantaneous frequencies, damping ratios. This is near the natural frequency ``0.12``, if the time step is assumed to be ``1.0``.
```julia
# for which frequency is the foliation calculated for
freq = 0.12/2/pi
# we did not specify sampling frequency, so we keep it as unit time-step
Tstep = 1.0

frequencyD, dampingD, amplitudeD = FoliationIdentify(dataIN, dataOUT, Tstep, opscal, "trivial", freq; orders = (P=7,Q=1,U=5,W=5), iterations = (f=1000, l=100))
```

Finally we plot the result.
```julia
pl = plot([omega, frequencyD[2:end]], [r, amplitudeD[2:end]], xlims=[0.11, 0.125], ylims=[0, 0.15], xlabel="frequency [rad/s]", ylabel="amplitude", label=["MAP" "DATA"])
display(pl)
```

!!! note
    The number of iterations are not high enough to produce accurate results, they are set so that the example runs quickly enough.
