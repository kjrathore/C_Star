using Pkg
Pkg.activate("env")

using KernelFunctions
using LinearAlgebra
using Distributions
using Plots
using BenchmarkTools
using LaTeXStrings
using Flux: Optimise
using Zygote
using Zygote:@adjoint   #added
using ComponentArrays, StableRNGs
using ProgressMeter
using JLD
rng = StableRNG(42)


# ---------------------------------------------------------------------
# Experiments on:
ITERS = 1400
LR = 0.3
out_dir = "$(pwd())/results/kernel"

println("iterations : ", ITERS)
println("Lr : ", LR)

# ---------------------------------------------------------------------


u0 = Float32[5.0; 0.0]
datasize = 1001

function tilman_system(u, datasize)
    x = zeros(datasize)
    i = zeros(datasize)
    x[1], i[1] = u         #inintial values
    r = 1.0
    c = collect(range(0.0, length=datasize, stop=4.0))
    K = 10.0
    h = 1.0
    T = 30.0
    # noises = rand(rng, Normal(0, 0.07), datasize)
    for k in 2:datasize
        eta = rand(Normal(0, 0.07), 1)[1]   #noises[k]
        x[k] =   r * x[k-1] * (1-x[k-1]/K) - c[k]* (x[k-1]^2 / (x[k-1]^2 + h^2)) + (1+i[k-1]) * x[k-1]
        i[k] = (1- (1/T)) * i[k-1] + eta
    end
    z = copy(transpose(cat(x, i, dims=2)))
    return z
end


data = Array(tilman_system(u0,datasize))

# data = zeros(size(X))
data[1,:] .= data[1,:] #./ (log(X[1, argmax(X[1,:])])- log(X[1,argmin(X[1,:])]))# X  #log_normalized
data[2,:] .= data[2,:] ./ (data[2, argmax(data[2,:])]- data[2,argmin(data[2,:])])   #normalized i noise  is very much important here.
# Plots.plot(transpose(data), labels=["X-state" "Noise"],  xlabel="Timesteps", ylabel="Abundance")# "C"])


x̄ = mean(data, dims=2)
noise_magnitude = 5e-2
data_n = data .+ (noise_magnitude * x̄) .* randn(rng, eltype(data), size(data))
data_n = transpose(data_n)

#  in our case we have only X train, 
X = data_n[begin:1000,:]
# # lets use di as y and check
y = diff(data_n[:,2])        #di



# # Plot simulation and noisy observations.
# plot(transpose(data), labels=[latexstring("env-state, \$x\$") latexstring("noise, \$i\$")], 
# xlabel="Timesteps", ylabel="Environmental State"; alpha=0.4, size=(700,400), margin=5Plots.mm)
# scatter!(1:1001, transpose(data_n); color=[1 2], label=[latexstring("Noisy Data, \$x\$") latexstring("Noisy Data, \$i\$")], markersize=3)

function relu(x)
    return max.(x, 0)
end


function kernel_creator(θ)
    return (exp(θ[1]) * SqExponentialKernel() + exp(θ[2]) * Matern32Kernel()) ∘
           ScaleTransform(exp(θ[3]))
end


function infer(x, xtrain, ytrain, θ)
    # input: x : vector from a matrix
            # xtrain: matrix
    x = RowVecs(x)
    k = kernel_creator(θ[1:3])
    x_df = RowVecs(xtrain)
    return kernelmatrix(k, x, x_df) *
           ((kernelmatrix(k, x_df) + exp(θ[4]) * I) \ ytrain)
end


function logistic_growth(u, parameters, states)
    # kernel approximator function; this will predict two dimensional array. [i,c]
        # consider inputing u [x, i] as single example 1D array.
    ut = u[1:1000, :]    #the current states
    di = infer(ut[:,1:2], X, y, parameters.θ)
    # println("di:",size(di))
    x_next = states.r .* ut[:,1] .* (1 .-ut[:,1] ./ states.K) .- ut[:,3] .* (ut[:,1].^2 ./ (ut[:,1].^2 .+ states.h .^2)) .+ (1 .+ ut[:,2]) .* ut[:,1]  
    # println("x_next: ", x_next)   
    u_next = hcat(x_next, ut[:,2].+di, relu(ut[:,3]))     #enforced positive C
    return u_next
end


# define initial parameters for kernel function:
θ = log.([1.1, 1.9, 0.9, 0.1])
#need to define x_test , x_train and y_train

# define parameters:
uhat = hcat(data_n, zeros(size(data_n)[1],1))      #transpose(vcat(data[begin:1000,:],zeros(1,size(X)[2])))
Parameters = ComponentArray(θ = θ, uhat=uhat)
states = (r=1.0, K=10.0, h=1.0, T=30.0)

println(Parameters.θ, size(Parameters.uhat))
# println("X, y:", size(X), size(y))


function loss(parameters)
    # here data contains only two state variables [x, i]; however parameters.uhat has [x, i, c]_t;   and bar_ut: [i, c]
    bar_ut = logistic_growth(parameters.uhat, parameters, states) 
    # println("size bar_ut",size(bar_ut))
    L_dyn = sum((parameters.uhat[2:end,:] .- bar_ut).^2)
    # println("dyn working")
    L_obs = sum((X .-parameters.uhat[1:1000,1:2]).^2)                     #observational Loss
    # println(L_dyn + L_obs)
    return L_dyn + L_obs
end


println("--------------------------------------------------------")
println("initial loss value: ", loss(Parameters))
println("----------------------Lets optimize now-----------------")

opt = Optimise.ADAM(LR)   #ADAGrad(LR)
@showprogress for iter in 1:ITERS
    grads = only((Zygote.gradient(loss, Parameters)))
    Optimise.update!(opt, Parameters, grads)
end

println("--------------------------------------------------------")
println("Latest loss value: ", loss(Parameters))
println("--------------------------------------------------------")


msd(a, b) = mean(abs2, a-b)


c_real = collect(range(0.0, 4.0, length=1001))[2:end]
c_hat = Parameters.uhat[2:end,3]

Plots.plot(c_real, labels=latexstring("true \$c\$"), xlabel="Timesteps", ylabel=latexstring("unresolved variable \$c\$"))
# Plots.scatter!(Parameters.uhat[:,3], labels=latexstring("predicted \$̂c\$"), markersize=2)
# pc = plot!(Parameters.uhat[:,3], color = "grey", linestyle = :dash, labels="")
pc = plot!(c_hat, labels=latexstring("predicted \$̂c\$"))
png(pc, "$(out_dir)/kernel_C_AdaMax_$(ITERS).png")


println("final parameters:θ =", Parameters.θ)
println("Mean Absolute test error: ",mean(abs.(c_real - c_hat)))
println("Root mean square deviation RMSD: ", sqrt(msd(c_real, c_hat)), "\n")


#Saving trained model parameters:
params = NamedTuple(Parameters)
# solution = NamedTuple(sol.u)
@save "./results/kernel_predicted_params.jld" params 

# iterations : 1400
# Lr : 0.3
# [0.09531017980432493, 0.6418538861723947, -0.10536051565782628, -2.3025850929940455](1001, 3)
# --------------------------------------------------------
# initial loss value: 1360.0329889607117
# ----------------------Lets optimize now-----------------
# Progress:  66%|█████████████████████████████              |  ETA: 0:10:58Progress:  66%|███████████████████████           | Progress:  82%  ETA:Progress: 100%|█████████████████████████████████| Time: 0:30:20
# --------------------------------------------------------
# Latest loss value: 25.698578725131917
# --------------------------------------------------------
# final parameters:θ =[-5.22384839555796, 2.7327486098558547, 2.2174783855339304, -3.514374439041028]
# Mean Absolute test error: 0.3869574379976306
# Root mean square deviation RMSD: 0.5298013068111774