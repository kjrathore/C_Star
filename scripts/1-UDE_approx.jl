# set the env first
using Pkg
Pkg.activate("env")

using SciMLBase
using Plots, Distributions, Random
using Optimization, OptimizationOptimisers
using LinearAlgebra, Statistics, LaTeXStrings
using Dates, CSV, DataFrames, JLD2
using ComponentArrays, Lux, Zygote,LineSearches, StableRNGs
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# Set a random seed for reproducible behaviour
@assert length(ARGS)>0 "Please provide seed as an argument";
seed = parse(Int64, ARGS[2]) #rand(1:1000)

rng = StableRNG(seed)   
train_size = parse(Int64, ARGS[1]) #training length [204, 530, 800]
# ---------------------------------------------------------------------
# Experiments on:
actf = celu
ITERS = 2000
lr = 0.03
out_dir = "$(pwd())/outputs/ude_celu2000/"
# mkdir("$(out_dir)")

println("------------------------------------")
println("Method: UDE")
println("Seed: ", seed)
println("Activation : ", actf)
println("Iterations : ", ITERS)
println("Lr : ", lr)

# ---------------------------------------------------------------------

#Data generation
u0 = Float32[10.0; 0.0]
datasize = 1001
c_real = collect(range(0.0, length=datasize, stop=4.0))

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
        x[k] = r * x[k-1] * (1-x[k-1]/K) - c[k]* (x[k-1]^2 / (x[k-1]^2 + h^2)) + (1+i[k-1]) * x[k-1]
        i[k] = (1- (1/T)) * i[k-1] + eta
    end
    z = copy(transpose(cat(x, i, dims=2)))
    return z
end

data = Array(tilman_system(u0,datasize))
data[1,:] .= data[1,:] #./ (log(X[1, argmax(X[1,:])])- log(X[1,argmin(X[1,:])]))# X  #log_normalized
data[2,:] .= data[2,:] ./ (data[2, argmax(data[2,:])]- data[2,argmin(data[2,:])])   
x̄ = mean(data, dims=2)
noise_magnitude = 5e-2
data_n = zeros(size(data))
data_n = data .+ (noise_magnitude * x̄) .* randn(rng, eltype(data), size(data))
data_n[1, :] .= max.(data_n[1,:], 0.0)   #eco-state should not be negative
X = data_n[:,1:train_size]
# println("size x", size(X))
#model arch:
NN = Lux.Chain(Lux.Dense(2, 8, actf),
            Lux.Dense(8, 16, actf),
            Lux.Dense(16, 16, actf), 
            Lux.Dense(16, 8, actf),
            Lux.Dense(8, 2))

# NN = Lux.Chain(Lux.Dense(2, 8, actf),
#             Lux.Dense(8, 16, actf),
#             Lux.Dense(16, 32, actf),
#             Lux.Dense(32, 32, actf), 
#             Lux.Dense(32, 8, actf),
#             Lux.Dense(8, 2))

# # Get the initial parameters and state variables of the model
NNparams, st = Lux.setup(rng, NN) #.|> device

uhat = vcat(X, rand(1,size(X)[2]))  #zeros(1,size(X)[2]))
# println("uhat ", size(uhat))
parameters = (NNparams = NNparams, uhat = uhat)
Parameters = ComponentArray(parameters)
states = (st=st, r=1.0, K=10.0, h=1.0)

function logistic_growth(u, parameters, states)
    u_hat = NN(u[1:2], parameters.NNparams, states.st)[1] # Network prediction
    x_next = states.r*u[1]*(1-u[1]/states.K)-u[3]*u[1]^2/(u[1]^2+states.h^2)+(1+u[2])*u[1]       #should be states.c*u[1]^2
    u = [x_next, u[2]+u_hat[1], relu(u[3]+u_hat[2])]     #enforced positive C
end



function loss(parameters, data)
    γ = 0.6
    L_dyn = 0
    for t in 2:(size(data)[2])
        bar_ut= logistic_growth(parameters.uhat[:,t-1], parameters, states)  #updated 1:2
        L_dyn += sum((parameters.uhat[:,t] .- bar_ut).^2)
    end
    L_obs = sum((data.-parameters.uhat[1:2,:]).^2)                     #observational Loss
    L_reg = sum(parameters.NNparams.layer_1.weight.^2)                 #Regularizational loss, penalizing the nueral networks parameters.
            + sum(parameters.NNparams.layer_2.weight.^2)
            + sum(parameters.NNparams.layer_3.weight.^2)
            + sum(parameters.NNparams.layer_4.weight.^2)
            + sum(parameters.NNparams.layer_5.weight.^2)
            # + sum(parameters.NNparams.layer_6.weight.^2)
    return γ * L_dyn + (1-γ) * L_obs + L_reg
end
	
	
t_start = now()
losses = Float64[]
callback = function (p, l; doplot = false)
    push!(losses, l)
    if length(losses) % 100 == 0
        println("Iteration: $(length(losses))  loss value: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x, X), adtype)
optprob = Optimization.OptimizationProblem(optf, Parameters)
sol1 = Optimization.solve(optprob, OptimizationOptimisers.ADAM(lr), 
                        callback = callback, maxiters = ITERS-500)

# start optimization 2:
optprob2 = Optimization.OptimizationProblem(optf, sol1.u)
sol2 = Optimization.solve(
        optprob2, LBFGS(linesearch = BackTracking()), 
        callback=callback, maxiters=500)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

t_end = now()
println("Opt time: ", canonicalize(t_end - t_start))

# ---------------------------------------------------------------------------

# pl = Plots.plot(log.(losses), label = ["Losses"], xlabel="iter", ylabel="Loss(log)")
# png(pl, "$(out_dir)/losses_$(lr)_$(actf)_$(ITERS).png")

# save model and predicted c
results = Dict()
push!(results, "seed" => seed)
push!(results, "lr" => lr)
push!(results, "time" =>  t_end - t_start)


true_c = c_real[2:train_size]
pred_c = sol2.u.uhat[3,2:end]

# println("c dims ", size(true_c), size(pred_c))
# Test Error
msd(a, b) = mean(abs2, a-b)
rmse(a, b) = sqrt(msd(a, b))
# push!(results, "maeC" => maeC)
push!(results, "rmseC" => rmse(true_c, pred_c))
println("C-RMSE : ", rmse(true_c, pred_c), "\n")

#only show x on plots
L = 21   #forecast length
#forecast in regime I 
I1 = train_size
preds = fill(NaN, 3,L)
preds[1:2,1] = data_n[:,I1]
preds[3,:] = c_real[I1:I1+L-1]

#since first value is 204...
for t in 2:(L)
    try
        preds[:, t] = logistic_growth(preds[:, t-1], sol2.u, states)  
    catch e
        println("Can not forecast further, since preds: ", preds[:, t-1])
        println("Error: ", e)
        break
    end
    # preds[:, t] = logistic_growth(preds[:, t-1], sol.u, states) 
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
@save "$(out_dir)/ude_predictions_$(train_size)_$(seed).jld2" predictions

# #Saving trained model parameters:
params = NamedTuple(Parameters)
solution = NamedTuple(sol2.u)
@save "$(out_dir)/ude_optimal_params_$(train_size)_$(seed).jld2" params solution

x_rmse = rmse(data[1, I1+1:I1+L-1], preds[1,2:end])
println("X-RMSE : ", x_rmse, "\n")
push!(results, "rmseX" => x_rmse)

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
append_to_csv("csv/rmseUDE$(train_size).csv", results_df)