# set the env first
using Pkg
Pkg.activate("env")

using SciMLBase
using Plots, Distributions, Random
using Optimization, OptimizationOptimisers
using LinearAlgebra, Statistics, LaTeXStrings
using Dates
using ComponentArrays, Lux, Zygote, StableRNGs
using JLD
# Set a random seed for reproducible behaviour
rng = StableRNG(42)

# ---------------------------------------------------------------------
# Experiments on:
actf = softsign
n_iters = 1500
lr = 0.03
out_dir = "$(pwd())/results/mse_loss"

println("Activation : ", actf)
println("iterations : ", n_iters)
println("Lr : ", lr)

# ---------------------------------------------------------------------

#Data generation
u0 = Float32[5.0; 0.0]
datasize = 1000

function tilman_system(u, datasize)

    x = zeros(datasize)
    i = zeros(datasize)
    x[1], i[1] = u         #inintial values
    r = 1.0
    c = collect(range(0.0, length=datasize, stop=4.0))
    # c = 2*sin.(collect(range(0.0, length=datasize, stop=10.0))) .+3.0
    K = 10.0
    h = 1.0
    T = 30.0
    # noises = rand(rng, Normal(0, 0.07), datasize)
    for k in 2:datasize
        eta = rand(rng, Normal(0, 0.07), 1)[1]   #noises[k]
        x[k] =   r * x[k-1] * (1-x[k-1]/K) - c[k]* (x[k-1]^2 / (x[k-1]^2 + h^2)) + (1+i[k-1]) * x[k-1]
        i[k] = (1- (1/T)) * i[k-1] + eta
    end
    z = copy(transpose(cat(x, i, dims=2)))
    return z
end

X = Array(tilman_system(u0,datasize))
data = zeros(size(X))
data[1,:] .= X[1,:] #./ (log(X[1, argmax(X[1,:])])- log(X[1,argmin(X[1,:])]))# X  #log_normalized
data[2,:] .= X[2,:] ./ (X[2, argmax(X[2,:])]- X[2,argmin(X[2,:])])   #normalized i noise  is very much important here.

#Plots.plot(transpose(data), labels=["X-state" "Noise"],  xlabel="Timesteps", ylabel="Abundance")# "C"])
x̄ = mean(X, dims=2)
noise_magnitude = 5e-2
data_n = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))


#model arch:
NN = Lux.Chain(Lux.Dense(2, 8, actf),
            Lux.Dense(8, 8, actf), 
            Lux.Dense(8, 8, actf),
            Lux.Dense(8, 2))

# # Get the initial parameters and state variables of the model
NNparams, st = Lux.setup(rng, NN) #.|> device

uhat =vcat(X,zeros(1,size(X)[2]))
parameters = (NNparams = NNparams, uhat = uhat)
Parameters = ComponentArray(parameters)

states = (st=st, r=1.0, K=10.0, h=1.0)

function logistic_growth(u, parameters, states)
    # println(size(u[1:2]))
    u_hat = NN(u[1:2], parameters.NNparams, states.st)[1] # Network prediction
    #lets do renormalization for specific instance only.
    x= u[1]  
    du1 = states.r*x*(1-x/states.K) - u[3]* x^2/(x^2 + states.h^2) + (1+u[2]) * x       #should be states.c*u[1]^2
    u = [du1, u[2]+u_hat[1], relu(u[3]+u_hat[2])]     #enforced positive C
end


# normed_1d(a,b) = abs(norm(a)-norm(b))/(norm(a)+norm(b))
# cosine_similarity(a, b) = dot(a,b) / (norm(a) * norm(b))
# cosine_distance(a, b) = (1 - cosine_similarity(a, b))/2

# function polar_loss(X, X̂, γ=0.4)
#     pl = sum(γ * normed_1d(X, X̂) +
#                (1-γ) * cosine_distance(X, X̂))
#     return  pl
# end


function loss(parameters, data)
    L_dyn = 0
    for t in 2:(size(data)[2])
        bar_ut= logistic_growth(parameters.uhat[:,t-1], parameters, states)  #updated 1:2
        L_dyn += sum(( parameters.uhat[:,t] .- bar_ut).^2)
    end
    L_obs = sum((data.-parameters.uhat[1:2,:]).^2)                     #observational Loss
    L_reg = sum(parameters.NNparams.layer_1.weight.^2)                 #Regularizational loss, penalizing the nueral networks parameters.
            + sum(parameters.NNparams.layer_2.weight.^2)
            + sum(parameters.NNparams.layer_3.weight.^2)
    return L_dyn + L_obs + L_reg
end
	
	
t_start = now()
losses = Float64[]
callback = function (p, l; doplot = false)
    # Optimisers.adjust!(p.original, 1 / p.iter)
    push!(losses, l)
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x, data_n), adtype)
optprob = Optimization.OptimizationProblem(optf, Parameters)

# step_size = 0.03; maxiter = 2000
sol = Optimization.solve(optprob, OptimizationOptimisers.ADAM(lr), 
                        callback = callback, maxiters = n_iters, progress=true)

t_end = now()
println("Optimization time: ",canonicalize(t_end - t_start))

pl = Plots.plot(log.(losses), color = :black, label = ["Losses"], xlabel="iterations", ylabel="Loss values (log)")
png(pl, "$(out_dir)/losses_$(lr)_$(actf)_$(n_iters).png")

p1 = Plots.plot(sol.u.uhat[1,:])
Plots.plot!(p1, sol.u.uhat[2,:])
p2 = Plots.scatter(data[1,:], sol.u.uhat[1,:])
Plots.plot!([0,1.0],[0,1.0], linestyle = :dash, color = "black")
p3 = Plots.scatter(data[2,:], sol.u.uhat[2,:])
Plots.plot!([0.0,1.0],[0.0,1.0], linestyle = :dash, color = "black")

# p4 = Plots.scatter(X[3,:], sol.u.uhat[3,:])
# Plots.plot!([0,1.0],[0,1.0], linestyle = :dash, color = "black") 
p22 = plot(p1,p2,p3)
png(p22, "$(out_dir)/preds_$(lr)_$(actf)_$(n_iters).png")

c_original = collect(range(0.0, length=datasize, stop=4.0))
true_c = c_original[2:end]
pred_c = sol.u.uhat[3,2:end]


msd(a, b) = mean(abs2, a-b)

# Test Error
println("Mean Absolute test error: ",mean(abs.(true_c - pred_c)))
println("Root mean square deviation RMSD: ", sqrt(msd(true_c, pred_c)), "\n")

p1 = Plots.plot(sol.u.uhat[3,2:end], labels=["C_hat"])
pc = Plots.plot!(p1, c_original[2:end], labels=["C"], xlabel="Timesteps", ylabel="Harvest rate (c) Values")   #after the first prediction
png(pc, "$(out_dir)/c_$(lr)_$(actf)_$(n_iters).png")

preds = zeros(3,20)
preds[:, 1] = sol.u.uhat[:,60]
preds[3,:] = c_original[60:79]

for t in 2:(20)
    bar_pred= logistic_growth(preds[:, t-1], sol.u, states) 
    preds[:, t] = bar_pred 
end

plot(transpose(data_n[:, begin:79]), labels=["env-state, x" "noise, i"])
fc = scatter!(60:79, transpose(preds), labels=[latexstring("\$̂x\$") latexstring("\$̂i\$") latexstring("\$c\$")])
png(fc, "$(out_dir)/forecst_$(lr)_$(actf)_$(n_iters).png")

tvals =  1.0:1.0:30.0 
u = [5.0, 0.007, 1.0]   #initial condition
p1 = Plots.plot(tvals, broadcast(x ->  logistic_growth(u, sol.u, states)[1], tvals), color = "grey", labels="C=1.0")
u = [5.0, 0.007, 2.0]   #initial condition
Plots.plot!(p1,tvals, broadcast(x -> logistic_growth(u, sol.u, states)[1], tvals), color = "grey", linestyle = :dash , labels="C=2.0")
u = [5.0, 0.007, 3.0]   #initial condition
xc = Plots.plot!(p1,tvals, broadcast(x -> logistic_growth(u, sol.u, states)[1], tvals), 
    color = "grey", linestyle = :dashdot , labels="C=3.0", xlabel="Time", ylabel="Env-state")
png(xc, "$(out_dir)/forecst_xc_$(lr)_$(actf)_$(n_iters).png");


#Saving trained model parameters:
params = NamedTuple(Parameters)
solution = NamedTuple(sol.u)
@save "./results/UDE_predicted_params.jld" params solution

# Activation : softsign
# iterations : 1500
# Lr : 0.03
# loss: 12.2: 100%|██████████████████████████| Time: 0:09:00
# Optimization time: 9 minutes, 38 seconds, 573 milliseconds
# Mean Absolute test error: 0.08438194847442301
# Root mean square deviation RMSD: 0.14169511967160364