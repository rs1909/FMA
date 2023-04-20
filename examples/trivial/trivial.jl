using FoliationsManifoldsAutoencoders
using Manifolds
using Plots

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

function generate(amp = 1.0)
    ndim = 4
    nruns = 32000
    npoints = 1
    xs = zeros(ndim, nruns*npoints)
    ys = zeros(ndim, nruns*npoints)
    aa = 2*rand(ndim,nruns) .- ones(ndim,nruns)
    ics =  aa ./ sqrt.(sum(aa.^2,dims=1)) .* (rand(1,nruns) .^ 0.7)
    u0 = zero(ics[:,1])
    for j=1:nruns
        u0 .= ics[:,j] * amp
        for k=1:npoints
            u1 = trivial(u0)
            xs[:,k+(j-1)*npoints] .= u0
            ys[:,k+(j-1)*npoints] .= u1
            u0 .= u1
        end
    end
    return xs, ys
end

MF = DensePolyManifold(4, 4, 5)
XF = fromFunction(MF, trivial)

MWt, XWt, MS, XS = iManifoldMAP(MF, XF, [3, 4], [])

amp_max = 0.15
opscal = ones(1,4)/4
That, Rhat_r = MAPFrequencyDamping(MWt, XWt, MS, XS, amp_max, output = opscal)
r = range(0, domain(That).right, length=1000)
omega = abs.(That.(r))
zeta = -log.(abs.(Rhat_r.(r))) ./ abs.(That.(r))

# create data
dataIN, dataOUT = generate(0.7)

# for which frequency is the foliation calculated for
freq = 0.12/2/pi
# we did not specify sampling frequency, so we keep it as unit time-step
Tstep = 1.0

frequencyD, dampingD, amplitudeD = FoliationIdentify(dataIN, dataOUT, Tstep, opscal, "trivial", freq; orders = (P=7,Q=1,U=7,W=7), iterations = (f=2000, l=600), node_rank = 32)

pl = plot([omega, frequencyD[2:end]], [r, amplitudeD[2:end]], xlims=[0.11, 0.125], ylims=[0, 0.15], xlabel="frequency [rad/s]", ylabel="amplitude", label=["MAP" "DATA"])
display(pl)
