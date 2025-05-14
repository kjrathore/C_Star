using Pkg
Pkg.activate("env")

using Distributions
using Plots
using Zygote
using Optim
using AbstractGPs, KernelFunctions
import AbstractGPs: mean_vector
using ParameterHandling
using ParameterHandling: flatten
using StableRNGs
using LaTeXStrings
using JLD2, CSV, DataFrames, Dates
using Flux: Optimise

# Set a random seed for reproducible behaviour
@assert length(ARGS)>0 "Please provide seed as an argument";
seed = parse(Int64, ARGS[2]) #rand(1:1000)
rng = StableRNG(seed)   
train_size = parse(Int64, ARGS[1]) #training length [204, 530, 800]
ITERS = 200

out_dir = "$(pwd())/outputs/gp_adam/"
# mkdir("$(out_dir)")

println("Method: GP")
println("Seed: ", seed)
println("Iterations : ", ITERS)

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
        x[k] = r * x[k-1] * (1-x[k-1]/K) - c[k]* (x[k-1]^2 / (x[k-1]^2 + h^2)) + (1+i[k-1]) * x[k-1]
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

X = data_n[begin:train_size,:]
y = diff(data_n[begin:train_size+1,2])        #target variable is the change in noise di
states = (r=1.0, K=10.0, h=1.0, T=30.0)

# #Initial parameter estimations
println("init_var ", var(y))
println("init lambda ", 0.2*(maximum(X) - minimum(X)))
# var noise   = β⋅var(target_data), β∈[0.01,0.1]
println("var noise ", 0.1*var(y))

# Declare model parameters using `ParameterHandling.jl` types.
flat_initial_params, unflatten = flatten((
    var_kernel = positive(var(y)),    #updated intial values
    λ = positive(0.2*(maximum(X) - minimum(X))),
    var_noise = positive(0.1*var(y)),
    uhat = hcat(X, rand(size(X,1),1)),
))


# # Construct a function to unpack flattened parameters and pull out the raw values.
unpack = ParameterHandling.value ∘ unflatten
params = unpack(flat_initial_params)

function construct_finite_gp(X, params)
    mn = x->0.
    kernel = params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ) #+ 1e-6 * I  # Add jitter
    return GP(mn, kernel)(RowVecs(X), params.var_noise)
end


# workaround needed for Zygote
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::ColVecs) = map(m.f, eachcol(x.X))
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::RowVecs) = map(m.f, eachrow(x.X))


function relu(x)
    return max.(x, 0)
end

# Adam optimizer configuration
opt = Optimise.ADAM(0.03)  # Set the learning rate

# Define the objective function
function objective(params)
    gp = construct_finite_gp(X, params)
    
    function infer(xtest, gp)
        post = posterior(gp, y)
        pred_y = marginals(post(RowVecs(xtest)))
        μ = mean.(pred_y)
        return μ
    end
    
    function logistic_growth(u)
        di = infer(u[:, 1:2], gp)
        x_next = states.r .* u[:,1].*(1 .-u[:,1]./states.K).-u[:,3].*(u[:,1].^2 ./(u[:,1].^2 .+ states.h .^2)) .+ (1 .+ u[:,2]) .* u[:,1]
        u_next = hcat(x_next, u[:,2].+di, relu(u[:,3]))
        return u_next
    end
    
    function loss(params)
        bar_ut = logistic_growth(params.uhat)
        L_dyn = sum((params.uhat .- bar_ut).^2)
        L_obs = sum((X .-params.uhat[:,1:2]).^2)
        lml = -logpdf(gp, y)
        return L_dyn + L_obs + lml
    end
    
    return loss(params)
end

# Training loop with Adam optimizer
println("--------------Started Optimizing-------------------")

flat_params = deepcopy(flat_initial_params)  # Initialize flattened parameters

t_start = now()
for epoch in 1:ITERS  # Set your desired number of epochs
    grads = Zygote.gradient(θ -> objective(unpack(θ)), flat_params)  # Compute gradients
    Optimise.update!(opt, flat_params, grads[1])  # Update parameters
    
    # Track progress
    loss_value = objective(unpack(flat_params))
    println("Epoch $epoch: Loss = $loss_value")
end

t_end = now()
println("Opt time: ", canonicalize(t_end - t_start))
println("--------------Completed Optimizing-----------------")

# @load "$(out_dir)/gp_optimal_params_$(train_size)_$(seed).jld2" params
final_params = unpack(flat_params)
# println("loaded params ",final_params )
#define mean sequared deviation
msd(a, b) = mean(abs2, a-b)
true_c = c_real[2:train_size]
pred_c = final_params.uhat[2:end,3]

results = Dict()
push!(results, "seed" => seed)
# push!(results, "lr" => LR)
# push!(results, "time" =>  t_end - t_start)

# Test Error
msd(a, b) = mean(abs2, a-b)
rmse(a, b) = sqrt(msd(a, b))

push!(results, "rmseC" => rmse(true_c, pred_c))
println("C-RMSE : ", rmse(true_c, pred_c), "\n")


#Saving trained model parameters:
params = NamedTuple(final_params)
@save "$(out_dir)/gp_optimal_params_$(train_size)_$(seed).jld2" params 


# ----------------------------------------------------------
# build GP with optimal params.
gp2 = construct_finite_gp(X, final_params)
# println("GP constructed")
posterior_gp = posterior(gp2, y)
# println("POST constructed")
# println("shape y", size(y))


function predict(xtest)
    pred_y = marginals(posterior_gp(RowVecs(xtest)))
    μ = mean.(pred_y)
    # println("mu ", μ)
    return μ
end
    

function logistic_growth_pred(u)
    # kernel approximator function; this will predict two dimensional array. [i,c]
        # consider inputing u [x, i] as single example 1D array.
    di = predict(reshape(u[1:2], (1,2)))    #will predict entire y vector
    x_next = states.r *u[1]*(1-u[1]/states.K)-u[3]*(u[1]^2 /(u[1]^2+states.h^2))+(1+u[2])*u[1]  
    # println("x_next: ", x_next)   
    u_next = hcat(x_next, u[2]+di[1], relu(u[3]))     #enforced positive C
    return u_next
end


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
        preds[:,t] = logistic_growth_pred(preds[:, t-1]) 
    catch e
        println("Can not forecast further, since preds: ", preds[:, t-1])
        println("Error: ", e)
        break
    end
    # preds[:, t] = logistic_growth_pred(preds[:, t-1]) 
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
@save "$(out_dir)/gp_predictions_$(train_size)_$(seed).jld2" predictions


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
append_to_csv("csv/rmseGP$(train_size).csv", results_df)

