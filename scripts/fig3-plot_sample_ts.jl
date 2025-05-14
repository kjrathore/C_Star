# set the env first
using Pkg
Pkg.activate("env")

using Distributions, Random, CSV, DataFrames
using Plots
using LaTeXStrings

rng = Random.seed!(42) #StableRNG(1111)

u0 = Float32[5.0; 0.0]
datasize = 1000

function tilman_system(u, datasize)
    x = zeros(datasize)
    i = zeros(datasize)
    x[1], i[1] = u         #inintial values
    r = 1.0
    c = collect(range(0.0, length=datasize, stop=4.0))
    K = 10.0
    h = 1.0
    T = 30.0
    noises = rand(rng, Normal(0, 0.07), datasize)
    for k in 2:datasize
        eta = noises[k]     #rand(rng, Normal(0, 0.07), 1)[1]   #noises[k]
        x[k] =  max(0,  r * x[k-1] * (1-x[k-1]/K) - c[k]* (x[k-1]^2 / (x[k-1]^2 + h^2)) + (1+i[k-1]) * x[k-1])
        i[k] = (1- (1/T)) * i[k-1] + eta
    end
    # z = copy(transpose(cat(x, i, dims=2)))
    return x
end

x = tilman_system(u0,datasize)
c = collect(range(0.0, length=datasize, stop=4.0))
# Plots
p = Plots.plot(hcat(x, c), labels=[L"x_t(c)" L"c(t)"],
               lc=[:black :gray],
               ls=[:solid :dash],
               lw=[1.3 2.4],
               xlabel="Time (t)", ylabel=L"Environmental state, $x_t(c)$",
               legend=:topright, legendfontsize=10,
               grid=false, size=(450, 300), 
               titlefontsize=12, labelfontsize=10,
               dpi=400, foreground_color_legend=nothing)
savefig(p, "../plots/x_timeseries.png")
