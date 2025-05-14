# lets focus on Regime I here:

using Pkg
Pkg.activate("env")

using JLD2
using Plots
using Statistics
using Lux
using KernelFunctions
using LinearAlgebra
using Distributions
# using BenchmarkTools
using LaTeXStrings
using AbstractGPs, KernelFunctions
import AbstractGPs: mean_vector
using ParameterHandling
using ParameterHandling: flatten
using ComponentArrays, StableRNGs
rng = StableRNG(86366)

# input arguments
H = 20   #forecast horizon
I1 = 204  #initial index for forecast
# I1 = 530
# I1 = 800

ude_p = JLD2.load("K:/CStar/outputs/ude/ude_optimal_params_$(I1)_86366.jld2")
kp = JLD2.load("K:/CStar/outputs/kernel/kernel_optimal_params_$(I1)_86366.jld2")
gp_params = JLD2.load("K:/CStar/outputs/gp_adam/gp_optimal_params_$(I1)_86366.jld2")
println("Model parameters loaded.")

# generate X#Data generation
u0 = Float32[10.0; 0.0]
datasize = 1001
c_real = collect(range(0.0, 4.0, length=datasize))

function tilman(u, datasize)
    x = zeros(datasize)
    i = zeros(datasize)
    x[1], i[1] = u         #inintial values
    r,K,h,T = 1.0,10.0,1.0,30.0
    c = collect(range(0.0, length=datasize, stop=4.0))
    for k in 2:datasize
        eta = rand(rng, Normal(0, 0.07), 1)[1]   #noises[k]
        x[k] = r * x[k-1] * (1-x[k-1]/K) - c[k]* (x[k-1]^2 / (x[k-1]^2 + h^2)) + (1+i[k-1]) * x[k-1]
        i[k] = (1- (1/T)) * i[k-1] + eta
    end
    z = copy(transpose(cat(x, i, dims=2)))
    return z
end

data = Array(tilman(u0,datasize))
data[1,:] .= data[1,:] #./ (log(X[1, argmax(X[1,:])])- log(X[1,argmin(X[1,:])]))# X  #log_normalized
data[2,:] .= data[2,:] ./ (data[2, argmax(data[2,:])]- data[2,argmin(data[2,:])]) 

x̄ = mean(data, dims=2)
noise_magnitude = 5e-2
data_n = data .+ (noise_magnitude * x̄) .* randn(rng, eltype(data), size(data))
data_n[1, :] .= max.(data_n[1,:], 0.0)
data_n = transpose(data_n)
#  in our case we have only X train, 
X = data_n[begin:1000,:]
y = diff(data_n[:,2])        #di


function get_intervals(forecasts)
    T = size(forecasts[1],1)
    mean_forecast = [mean([forecast[t] for forecast in forecasts]) for t in 1:T]
    # Calculate standard deviation at each time point
    std_forecast = [std([forecast[t] for forecast in forecasts]) for t in 1:T]
    # Define confidence interval (e.g., 95% confidence interval with ±1.96 * std)
    confidence_level = 1.96
    upper_interval = mean_forecast .+ confidence_level .* std_forecast
    lower_interval = mean_forecast .- confidence_level .* std_forecast
    return upper_interval, lower_interval, mean_forecast
end


actf = celu
#model arch:
NN = Lux.Chain(Lux.Dense(2, 8, actf),
            Lux.Dense(8, 16, actf),
            Lux.Dense(16, 16, actf), 
            Lux.Dense(16, 8, actf),
            Lux.Dense(8, 2))

# # Get the initial parameters and state variables of the model
NNparams, st = Lux.setup(rng, NN) #.|> device
states = (st=st, r=1.0, K=10.0, h=1.0)

function ude_prediction(u, parameters, states)
    η = rand(Normal(0, 0.07), 1)[1]
    u[2] = u[2]+η    #update noise 
    u_hat = NN(u[1:2], parameters.NNparams, states.st)[1] # Network prediction
    x_next = states.r*u[1]*(1-u[1]/states.K) - u[3]* u[1]^2/(u[1]^2 + states.h^2) + (1+u[2]) * u[1]       #should be states.c*u[1]^2
    return [x_next, u[2]+u_hat[1], relu(u[3]+u_hat[2])]     #enforced positive C
end

# ude_prediction(ude_p["solution"].uhat[:,1], ude_p["solution"], states)
# new_c = collect(range(1.0, length=20, stop=2.0))

function forecast_ude_x()
    preds = zeros(3,H)
    preds[:,1] = [data[1,I1],data[2, I1],0.0] 
    preds[3,:] = c_real[I1:I1+H-1]   #using real values of c
    # lets pass original c values manually.
    for t in 2:(H)
        # x_n, i_n = ude_prediction(preds[:, t-1],ude_p, states)
        # preds[1:2, t] = [x_n, i_n]
        preds[:,t] = ude_prediction(preds[:, t-1], ude_p["solution"], states)
    end
    return preds[1,:]   #return only x_vector forecast
end


function get_forecasts()
    forecasts = [forecast_ude_x() for run in 1:50]
    return forecasts
end

println("UDE Forecasts:")
ude_forecasts = get_forecasts()
upr_ude, lwr_ude, mean_ude = get_intervals(ude_forecasts)


# --------------------------------------------------------------------------


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

states = (r=1.0, K=10.0, h=1.0, T=30.0)

function kernel_predictions(u, parameters, states)
    # kernel approximator function; this will predict two dimensional array. [i,c]
        # consider inputing u [x, i] as single example 1D array.
    ut = u
    η = rand(Normal(0, 0.07), 1)[1]
    ut[2] = ut[2]+η 
    # println("ut", ut, reshape(ut, (1,2)))
    di = infer(reshape(ut[1:2], (1,2)), X, y, parameters.θ)
    # di = infer(reshape(ut[1:2], (1,2)), X, y, parameters.θ)
    x_next = states.r *ut[1]*(1-ut[1]/states.K)-ut[3]*(ut[1]^2 /(ut[1]^2+states.h^2))+(1+ut[2])*ut[1]  
    # u_next = hcat(x_next, ut[2]+di[1], relu(ut[3]))     #enforced positive C
    return x_next, ut[2]+di[1]
end



function forecast_kernel_x()
    preds = zeros(3,H)
    preds[:,1] = [data[1,I1],data[2, I1],0.0] #nnsol.uhat[:,I1]
    preds[3,:] = c_real[I1:I1+H-1]
    # lets pass original c values manually.
    for t in 2:(H)
        x_n, i_n = kernel_predictions(preds[:, t-1], kp["params"], states)
        preds[1:2, t] = [x_n, i_n]
    end
    return preds[1,:]   #retrun only x_vector forecast
end


function get_forecasts()
    forecasts = [forecast_kernel_x() for run in 1:50]
    return forecasts
end

println("Kernel Forecasts:")
kernel_forecasts = get_forecasts()
upr_k, lwr_k,mean_k = get_intervals(kernel_forecasts)
# --------------------------------------------------------------------------



# Declare model parameters using `ParameterHandling.jl` types.
flat_initial_params, unflatten = flatten((
    var_kernel = positive(0.6),
    λ = positive(2.5),
    var_noise = positive(0.1),
    uhat = hcat(data_n, rand(size(data_n)[1],1)),
))

# Construct a function to unpack flattened parameters and pull out the raw values.
unpack = ParameterHandling.value ∘ unflatten
params = unpack(flat_initial_params)

function construct_finite_gp(X, params)
    # println(params)
    mn = x->0.
    kernel = params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ)
    return GP(mn, kernel)(RowVecs(X), params.var_noise)
end


gp2 = construct_finite_gp(X, gp_params["params"])
post = posterior(gp2, y)
# println("shape y", size(y))

function predict(xtest)
    pred_y = marginals(post(RowVecs(xtest)))
    μ = mean.(pred_y)
    return μ
end
    
function gaussian_prediction(u)
    # kernel approximator function; this will predict two dimensional array. [i,c]
        # consider inputing u [x, i] as single example 1D array.
    η = rand(Normal(0, 0.07), 1)[1]
    u[2] = u[2]+η 
    di = predict(reshape(u[1:2], (1,2)))    #will predict entire y vector
    x_next = states.r *u[1]*(1-u[1]/states.K)-u[3]*(u[1]^2 /(u[1]^2+states.h^2))+(1+u[2])*u[1]  
    # u_next = hcat(x_next, u[2]+di[1], relu(u[3]))     
    return x_next, u[2]+di[1]
end


function forecast_gaussian_x()

    preds = zeros(3,H)
    preds[:,1] = [data[1,I1],data[2, I1],0.0] #nnsol.uhat[:,I1]
    preds[3,:] = c_real[I1:I1+H-1]
    # lets pass original c values manually.
    for t in 2:(H)
        x_n, i_n = gaussian_prediction(preds[:, t-1])
        preds[1:2, t] = [x_n, i_n]
    end
    return preds[1,:]   #return only x_vector forecast
end


function get_forecasts()
    forecasts = [forecast_gaussian_x() for run in 1:50]
    return forecasts
end

println("Gaussian Forecasts:")
gaussian_forecasts = get_forecasts()
upr_g, lwr_g, mean_g = get_intervals(gaussian_forecasts)

# --------------------------------------------------------------------------
# Plotting
gr(legendfontsize=16,
        markerstrokewidth=0, 
        xtickfontsize=16, ytickfontsize=16, 
        xguidefontsize=16, yguidefontsize=16,
        dpi=400,
        yticks=[0,5,10,15,18])


plot_array = Any[]
H2 = I1+H-1
I0 = I1-10
# Kernel 
plot(I0:H2,data[1,I0:H2],  color=:black, lw=2, labels=L"x")
plot!(I1:H2, mean_k, seriestype=:line, marker=:utriangle,lw=2, color="#AA3377", label="DE-Kernel")
plot_k = plot!(I1:H2, lwr_k, fillrange= upr_k, label="Forecast\nInterval", color="#AA3377", alpha=0.2)#, ylim=(0, 17))
push!(plot_array, plot_k)

# Guassian
plot(I0:H2,data[1,I0:H2],  color=:black, lw=2, labels="")
plot!(I1:H2, mean_g, seriestype=:line, marker=:circle, lw=2, color="#4477AA", label="DE-GP")
plot_g = plot!(I1:H2, lwr_g, fillrange= upr_g, label="Forecast\nInterval", color="#4477AA", alpha=0.2)#, ylim=(0, 17))
push!(plot_array, plot_g)

# UDE
plot(I0:H2,data[1,I0:H2],  color=:black, lw=2, labels="")
plot!(I1:H2, mean_ude,seriestype=:line, marker=:rect,lw=2, color="#228833", label="UDE")
plot_u = plot!(I1:H2, lwr_ude, fillrange= upr_ude, label="Forecast\nInterval", 
        xlabel="Time", color="#228833", alpha=0.2)
push!(plot_array, plot_u)

allp = plot!(plot_array..., layout = (3,1), 
        size=(620, 800),   #(820, 800)
        ylabel = "Environmental\nstate",
        margin=3Plots.mm, link=:all, 
        legend=false,
        titleloc = :left, titlefontsize=16,
        right_margin=4Plots.mm,
        # legend=:outerright,
        foreground_color_legend=nothing,
        grid=false)

# ylabel!(allp, L"Environmental states, $x$")
# annotate!((0.1, 0.5), text(L"Environmental states, $x$", :left, 12, rotation = 90))

png(allp, "../plots/forecast_CI_$(I1)_3x1.png")