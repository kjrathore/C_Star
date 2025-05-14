# set the env first
using Pkg
Pkg.activate("env")

using StateSpaceModels, Random, StatsBase
using LaTeXStrings, Distributions, StableRNGs
using Dates, CSV, DataFrames

# Set a random seed for reproducible behaviour
@assert length(ARGS)>0 "Please provide seed as an argument";
seed = parse(Int64, ARGS[2]) #rand(1:1000)

rng = StableRNG(seed)   
train_size = parse(Int64, ARGS[1]) #training length [204, 530, 800]
# ---------------------------------------------------------------------
# Experiments on:
# actf = celu
# ITERS = 2000
# lr = 0.03
out_dir = "$(pwd())/outputs/arima/"
# mkdir("$(out_dir)")

println("------------------------------------")
println("Method: SSMs")
println("Seed: ", seed)
# println("Activation : ", actf)
# println("Iterations : ", ITERS)
# println("Lr : ", lr)

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

steps_ahead = 20
x_current = data_n[1,1:train_size]

# save model and predicted c
results = Dict()
push!(results, "seed" => seed)

function get_rmsd(x, forec)
    x̂ = reduce(vcat, forec.expected_value)
    return StatsBase.rmsd(x, x̂)
end

# # SARIMA
# t_start = now()
# model_sarima = StateSpaceModels.SARIMA(x_current; order = (0, 1, 1), seasonal_order = (0, 1, 1, 20))
# StateSpaceModels.fit!(model_sarima, save_hyperparameter_distribution=false)
# t_end = now()
# push!(results, "SARIMA_time" =>  t_end - t_start)
# forec_sarima = forecast(model_sarima, steps_ahead)
# push!(results, "SARIMA_rmse" => get_rmsd(data[1,train_size+1:train_size+steps_ahead], forec_sarima))
# println("SARIMA forecast successful.")

# # Unobserved Components
# t_start = now()
# model_uc = StateSpaceModels.UnobservedComponents(x_current; trend = "local linear trend", seasonal = "stochastic 20")
# StateSpaceModels.fit!(model_uc,  save_hyperparameter_distribution=false)
# t_end = now()
# push!(results, "UC_time" =>  t_end - t_start)
# forec_uc = forecast(model_uc, steps_ahead)
# push!(results, "UC_rmse" => get_rmsd(data[1,train_size+1:train_size+steps_ahead], forec_uc))
# println("UC forecast successful.")

# # Exponential Smoothing
# t_start = now()
# model_ets = StateSpaceModels.auto_ets(x_current; seasonal = 20)
# StateSpaceModels.fit!(model_ets, save_hyperparameter_distribution=false)
# t_end = now()
# push!(results, "ETS_time" =>  t_end - t_start)
# forec_ets = forecast(model_ets, steps_ahead)
# push!(results, "ETS_rmse" => get_rmsd(data[1,train_size+1:train_size+steps_ahead], forec_ets))
# println("ETS forecast successful.")

# # Naive model
# t_start = now()
# model_naive = StateSpaceModels.SeasonalNaive(x_current, 20)
# StateSpaceModels.fit!(model_naive)
# t_end = now()
# push!(results, "Naive_time" =>  t_end - t_start)
# forec_naive = forecast(model_naive, steps_ahead)
# push!(results, "Naive_rmse" => get_rmsd(data[1,train_size+1:train_size+steps_ahead], forec_naive))
# println("NAIVE forecast successful.")

#AR model
t_start = now()
model_ar = StateSpaceModels.auto_arima(x_current; seasonal=20)
StateSpaceModels.fit!(model_ar, save_hyperparameter_distribution=false)
t_end = now()
push!(results, "AutoAR_time" =>  t_end - t_start)
forec_ar = forecast(model_ar, steps_ahead)
x_future = vcat(forec_ar.expected_value...)
# push!(results, "AutoAR_rmse" => get_rmsd(data[1,train_size+1:train_size+steps_ahead], forec_ar))
println("AUTO_AR forecast successful.")
# ---------------------------------------------------------------------------


#evaluate:
msd(a, b) = mean(abs2, a-b)
rmsd(a, b) = sqrt(msd(a, b))

true_x = data[train_size+1:train_size+steps_ahead]
rmse1 = rmsd(true_x[1], x_future[1])
rmse5 = rmsd(true_x[1:5], x_future[1:5])
rmse10 = rmsd(true_x[1:10], x_future[1:10])
rmse20 = rmsd(true_x, x_future)

println("RMSE-1: \t", rmse1)
println("RMSE-5: \t", rmse5)
println("RMSE-10: \t", rmse10)
println("RMSE-20: \t", rmse20)
push!(results, "rmse1" =>  rmse1)
push!(results, "rmse5" =>  rmse5)
push!(results, "rmse10" =>  rmse10)
push!(results, "rmse20" =>  rmse20)
push!(results, "method" =>  "ARIMA")


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

#save real value and forecasted values of x
results_df = DataFrame(results)
# CSV.write("$(out_dir)/results_$(train_size)_$(seed).csv", results_df)
# Read the CSV file into a DataFrame
append_to_csv("csv/rmseARIMA$(train_size).csv", results_df)
