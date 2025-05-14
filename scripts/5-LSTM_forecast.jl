# set the env first
using Pkg
Pkg.activate("env")

using Plots, Distributions, Random
using LaTeXStrings
using Dates, StableRNGs, Flux
using Flux: throttle

using StatsBase, CSV, DataFrames

# Set a random seed for reproducible behaviour
@assert length(ARGS)>0 "Please provide seed as an argument";
seed = parse(Int64, ARGS[2]) #rand(1:1000)

rng = StableRNG(seed)   
train_size = parse(Int64, ARGS[1]) #training length [204, 530, 800]
# ---------------------------------------------------------------------
# Experiments on:
actf = celu
ITERS = 1500
lr = 0.03
out_dir = "$(pwd())/outputs/lstm/"
# mkdir("$(out_dir)")

println("------------------------------------")
println("Method: SSMs")
println("Seed: ", seed)
println("Activation : ", actf)
println("Iterations : ", ITERS)
println("Lr : ", lr)

# ---------------------------------------------------------------------

u0 = Float32[10.0; 0.0]
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
        eta = rand(rng, Normal(0, 0.07), 1)[1]   #noises[k]
        x[k] =   r * x[k-1] * (1-x[k-1]/K) - c[k]* (x[k-1]^2 / (x[k-1]^2 + h^2)) + (1+i[k-1]) * x[k-1]
        i[k] = (1- (1/T)) * i[k-1] + eta
    end
    # copy(transpose(cat(x, i, dims=2)))
    return Array(x)
end

data = tilman_system(u0,datasize)
# Normalize the data
# data_n = data ./ (maximum(data) - minimum(data))   #(data .- mean_val) ./ std_val

x̄ = mean(data, dims=2)
noise_magnitude = 5e-2
data_n = data .+ (noise_magnitude * x̄) .* randn(rng, eltype(data), size(data))
data_n .= max.(data_n, 0.0)   #eco-state should not be negative

data_t = Float32.(data_n[1:train_size])

# Create input sequences (e.g., length 10) and targets
seq_len = 10
X = [data_t[i:i+seq_len-1] for i in 1:(length(data_t) - seq_len)]
y = [data_t[i+seq_len] for i in 1:(length(data_t) - seq_len)]

# Convert to arrays
X = hcat(X...)'
y = reshape(y, 1, :)

# Define the LSTM model
input_size = seq_len
output_size = 1

model = Chain(
    LSTM(input_size, 20),       # LSTM layer
    Dense(20, 10, celu),
    Dense(10, 10, celu),
    Dense(10, output_size)     # Fully connected layer
)

loss_fn = (x, y) -> Flux.mse(model(x), y)
optimizer = Flux.Optimise.Adam(0.03)


# Prepare data for training (LSTM expects inputs as 3D tensors)
X_train = reshape(X, size(X, 2), size(X, 1), 1)
y_train = reshape(y, (1, size(y,2), 1))
# println(size(X_train), size(y_train))

# save model and predicted c
results = Dict()
push!(results, "seed" => seed)

t_start = now()
# Training loop
for epoch in 1:ITERS
    Flux.train!(loss_fn, Flux.params(model), [(X_train, y_train)], optimizer)
    if epoch % 100 == 0
        println("Epoch $epoch: Loss = $(loss_fn(X_train, y_train))")
    end
end
t_end = now()
push!(results, "opt_time" =>  t_end - t_start)
println("LSTM model trained.")

#this is to be done before prediction.
Flux.reset!(model)


function forecast(model, initial_seq, steps)
    forecasted = Float32[]  # Match the data type with the model
    seq = Float32.(initial_seq)  # Convert initial sequence to Float32

    for _ in 1:steps
        # Reshape sequence to 3D tensor: (seq_len, 1, 1)
        input_tensor = reshape(seq, (length(seq), 1, 1))
        # Predict the next value
        next_values = reshape(model(input_tensor), :)  # Flatten the output
        next_value_scalar = next_values[1]  # Extract scalar value
        # Append the prediction to the forecast
        append!(forecasted, next_value_scalar)
        # Shift and append the new value to the sequence
        seq = vcat(seq[2:end], next_value_scalar)
    end

    return forecasted
end

# Example usage
initial_seq = data_t[end-seq_len+1:end]  # Ensure Float32 compatibility
future_steps = 20
forecast_x = forecast(model, initial_seq, future_steps)


#evaluate:
msd(a, b) = mean(abs2, a-b)
rmsd(a, b) = sqrt(msd(a, b))

# # Test Error
println("RMSE: \t", )

true_x = data[train_size+1:train_size+future_steps]
rmse1 = rmsd(true_x[1], forecast_x[1])
rmse5 = rmsd(true_x[1:5], forecast_x[1:5])
rmse10 = rmsd(true_x[1:10], forecast_x[1:10])
rmse20 = rmsd(true_x, forecast_x)

println("RMSE-1: \t", rmse1)
println("RMSE-5: \t", rmse5)
println("RMSE-10: \t", rmse10)
println("RMSE-20: \t", rmse20)
push!(results, "rmse1" =>  rmse1)
push!(results, "rmse5" =>  rmse5)
push!(results, "rmse10" =>  rmse10)
push!(results, "rmse20" =>  rmse20)
push!(results, "method" =>  "LSTM")


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
append_to_csv("csv/rmseLSTM$(train_size).csv", results_df)
