using Pkg
Pkg.activate("env")

using KernelFunctions
using LinearAlgebra
using Distributions
using Plots
# using BenchmarkTools
using LaTeXStrings
using Flux: Optimise
using Zygote
using Zygote:@adjoint   #added
using ComponentArrays, StableRNGs
using ProgressMeter
using JLD2, Dates, CSV, DataFrames
using Random: seed!
# rng = StableRNG(42)

# Set a random seed for reproducible behaviour
@assert length(ARGS)>0 "Please provide seed as an argument";
seed = parse(Int64, ARGS[2]) #rand(1:1000)
rng = StableRNG(seed)   
train_size = parse(Int64, ARGS[1]) #training length [204, 530, 800]
# ---------------------------------------------------------------------
# Experiments on:
ITERS = 10
LR = 0.03
out_dir = "$(pwd())/outputs/kernel/"
# mkdir("$(out_dir)")

println("Method: Kernel")
println("Seed: ", seed)
println("Iterations : ", ITERS)
println("Lr : ", LR)

# ---------------------------------------------------------------------


u0 = Float32[10.0; 0.0]
datasize = 1001
c_real = collect(range(0.0, length=datasize, stop=4.0))

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
        eta = rand(rng, Normal(0, 0.07), 1)[1]   #noises[k]
        x[k] =   r * x[k-1] * (1-x[k-1]/K) - c[k]* (x[k-1]^2 / (x[k-1]^2 + h^2)) + (1+i[k-1]) * x[k-1]
        i[k] = (1- (1/T)) * i[k-1] + eta
    end
    z = copy(transpose(cat(x, i, dims=2)))
    return z
end


data = Array(tilman_system(u0,datasize))
data[1,:] .= data[1,:] #./ (log(X[1, argmax(X[1,:])])- log(X[1,argmin(X[1,:])]))# X  #log_normalized
data[2,:] .= data[2,:] ./ (data[2, argmax(data[2,:])]- data[2,argmin(data[2,:])])   #normalized i noise  is very much important here.
# Plots.plot(transpose(data), labels=["X-state" "Noise"],  xlabel="Timesteps", ylabel="Abundance")# "C"])


x̄ = mean(data, dims=2)
noise_magnitude = 5e-2
data_n = data .+ (noise_magnitude * x̄) .* randn(rng, eltype(data), size(data))
# Ensure noisy observations for environmental states must remain >= 0.
data_n[1, :] .= max.(data_n[1,:], 0.0)
data_n = transpose(data_n)

#  in our case we have only X train, 
X = data_n[begin:train_size,:]
y = diff(data_n[begin:train_size+1,2])        #di #target variable is the change in noise di


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
    # println("u,X,y",size(u), size(X), size(y))
    ut = u[1:train_size, :]    #the current states
    di = infer(ut[:,1:2], X, y, parameters.θ)
    x_next = states.r .* ut[:,1] .* (1 .-ut[:,1] ./ states.K) .- ut[:,3] .* (ut[:,1].^2 ./ (ut[:,1].^2 .+ states.h .^2)) .+ (1 .+ ut[:,2]) .* ut[:,1]  
    # println("x_next: ", x_next)   
    u_next = hcat(x_next, ut[:,2].+di, relu(ut[:,3]))     #enforced positive C
    return u_next
end


# define initial parameters for kernel function:
# Assuming x (input data) and y (target data) are given
data_var = var(y)
x_range = maximum(X) - minimum(X)

# Initial parameter estimates
θ1 = log(data_var / 2)  # Variance for SqExponentialKernel
θ2 = log(data_var / 2)  # Variance for Matern32Kernel
θ3 = log(1 / x_range)   # Scale (lengthscale)
# θ = log.([1.1, 1.9, 0.9, 0.1])
θ = [θ1, θ2, θ3, 2.5]
#need to define x_test , x_train and y_train

# define parameters: rand(1,size(X)[2])
uhat = hcat(X, rand(size(X,1),1))
# println("uhat.shape:", size(uhat))     
Parameters = ComponentArray(θ = θ, uhat=uhat)
states = (r=1.0, K=10.0, h=1.0, T=30.0)

function loss(parameters)
    # here data contains only two state variables [x, i]; however parameters.uhat has [x, i, c]_t;   and bar_ut: [i, c]
    bar_ut = logistic_growth(parameters.uhat, parameters, states) 
    # println("uhat",size(parameters.uhat), "bar_ut", size(bar_ut))
    L_dyn = sum((parameters.uhat .- bar_ut).^2)
    L_obs = sum((X .-parameters.uhat[:,1:2]).^2)                     #observational Loss
    # println(L_dyn + L_obs)
    return L_dyn + L_obs
end

t_start = now()
println("--------------------------------------------------------")
println("Inital parameters, θ ", θ)
println("Initial loss value: ", loss(Parameters))
println("----------------------Lets optimize now-----------------")

opt = Optimise.ADAM(LR)   #ADAGrad(LR)
for iter in 1:ITERS
    grads = only((Zygote.gradient(loss, Parameters)))
    Optimise.update!(opt, Parameters, grads)
end

t_end = now()
# println("--------------------------------------------------------")
println("Opt time: ", canonicalize(t_end - t_start))
println("Latest loss value: ", loss(Parameters))
println("--------------------------------------------------------")


msd(a, b) = mean(abs2, a-b)
# c_original = collect(range(0.0, 4.0, length=1001))
# true_c = collect(range(0.0, 4.0, length=1001))[2:end]
pred_c = Parameters.uhat[2:end,3]

# Plots.plot(true_c, labels=L"true, c", xlabel="Timesteps", ylabel=L"unobserved variable ̂c")
# Plots.scatter!(Parameters.uhat[:,3], labels=latexstring("predicted \$̂c\$"), markersize=2)
# pc = plot!(Parameters.uhat[:,3], color = "grey", linestyle = :dash, labels="")
# pc = plot!(pred_c, labels=L"predicted, ̂c")
# png(pc, "$(out_dir)/kernel_C_AdaMax_$(ITERS).png")


println("final parameters:θ = ", Parameters.θ)

results = Dict()
push!(results, "seed" => seed)
push!(results, "lr" => LR)
push!(results, "time" =>  t_end - t_start)

# Test Error
msd(a, b) = mean(abs2, a-b)
rmse(a, b) = sqrt(msd(a, b))
# maeC = mean(abs.(true_c - pred_c))
# push!(results, "maeC" => maeC)
true_c = c_real[2:train_size]
push!(results, "rmseC" => rmse(true_c, pred_c))
# println("C-MAE : ",maeC)
println("C-RMSE : ", rmse(true_c, pred_c), "\n")


# #Saving trained model parameters:
params = NamedTuple(Parameters)
@save "$(out_dir)/kernel_optimal_params_$(train_size)_$(seed).jld2" params 


# ----------------------------------------------------------

function logistic_growth_pred(u, parameters, states)
    # kernel approximator function; this will predict two dimensional array. [i,c]
        # consider inputing u [x, i] as single example 1D array.
    ut = u#[1:1000, :]    #ut is 1D here
    # println("ut",reshape(ut[1:2], (1,2)))
    di = infer(reshape(ut[1:2], (1,2)), X, y, parameters.θ)
    # println("di:",size(di))
    x_next = states.r *ut[1]*(1-ut[1]/states.K)-ut[3]*(ut[1]^2 /(ut[1]^2+states.h^2))+(1+ut[2])*ut[1]  
    # println("x_next: ", x_next)   
    u_next = hcat(x_next, ut[2]+di[1], relu(ut[3]))     #enforced positive C
    return u_next
end

println("Forecasting now..")
#only show x on plots
L = 21   #forecast length
#forecast in regime I 
I1 = train_size
preds = fill(NaN, 3,L)
preds[1:2,1] = data_n[I1,:]
preds[3,:] = c_real[I1:I1+L-1]

#since first value is 204...
for t in 2:(L)
    try
        preds[:,t] = logistic_growth_pred(preds[:, t-1], Parameters, states) 
    catch e
        println("Can not forecast further, since preds: ", preds[:, t-1])
        println("Error: ", e)
        break
    end
    # preds[:, t] = logistic_growth_pred(preds[:, t-1], Parameters, states) 
end

# Check if array has any NaN
if any(isnan, preds)
    println("Forecasts failed. has NaN values")
else
    println("Forecast successful!!")
end
    
# save x 
predictions = (true_c = true_c, 
            pred_c = pred_c, 
            true_x = data[1, I1:I1+L-1],
            preds = preds)
@save "$(out_dir)/kernel_predictions_$(train_size)_$(seed).jld2" predictions

function append_to_csv(file_path::String, data::DataFrame)
    # Check if the file exists
    if isfile(file_path)
        # Append data to the existing file
        CSV.write(file_path, data, append=true)
    else
        # Create a new file and write data
        CSV.write(file_path, data)
    end
end
# save real value and forecasted values of x
results_df = DataFrame(results)
append_to_csv("csv/rmseKERNEL$(train_size).csv", results_df)