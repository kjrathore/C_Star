using Pkg
Pkg.activate("C:/Users/kj_ra/Box/Ecology/Project_CStar/HPC_codes/discrete_dym/env_c")

using JLD2, Plots
using LaTeXStrings
using ComponentArrays, StableRNGs
using Distributions
# using PyPlot
using JSON
using StatsPlots
using DataFrames, CSV, Glob
using HypothesisTests
# read random seeds from file
f = open("rnds.txt", "r")
rnds = []
for lines in readlines(f)
    push!(rnds, lines)
end


#  lets compute Tilman simulated data again.
#Data generation

function tilman_system(seed)
    datasize = 1001
    x, i = zeros(datasize), zeros(datasize)
    x[1], i[1] = 10.0, 0.0         #inintial values
    c_real = collect(range(0.0, length=datasize, stop=4.0))
    r, K, h, T = 1.0, 10.0, 1.0, 30 
    c = collect(range(0.0, length=datasize, stop=4.0))
    rng = StableRNG(seed)   
    for k in 2:datasize
        eta = rand(rng, Normal(0, 0.07), 1)[1]   #noises[k]
        x[k] = r * x[k-1] * (1-x[k-1]/K) - c[k]* (x[k-1]^2 / (x[k-1]^2 + h^2)) + (1+i[k-1]) * x[k-1]
        i[k] = (1- (1/T)) * i[k-1] + eta
    end
    TS = transpose(cat(x, i, dims=2))
    data = zeros(2, size(TS,2))
    data[1,:] .= TS[1,:] #./ (log(X[1, argmax(X[1,:])])- log(X[1,argmin(X[1,:])]))# X  #log_normalized
    data[2,:] .= TS[2,:] ./ (TS[2, argmax(TS[2,:])]- TS[2,argmin(TS[2,:])])   #normalized i noise  is very much important here.
    x̄ = mean(x)
    noise_magnitude = 5e-2
    X_n = data[1,:] .+ (noise_magnitude * x̄).*randn(rng, eltype(data[1,:]), size(data[1,:]))
    # X_n = data .+ (noise_magnitude * x̄) .* randn(rng, eltype(data), size(data))
    X_n .= max.(X_n, 0.0)
    return TS[1,:], X_n
end

# X, X_n = tilman_system(97004)
# plot(X, label="X")
# scatter!(X_n, markersize=2, label="noisy data")   #lets go with this one


function rmse(y_true, y_pred)
    return sqrt(mean((y_true .- y_pred).^2))
end

function eval_forecast(true_xs, pred_xs)
    rmse1, rmse5, rmse10, rmse20 = [], [], [], []
    true_x = true_xs[2:end]         #leave first prediction out.
    pred_x = pred_xs[2:end]     #removed first
    rmse1 = rmse(true_x[1], pred_x[1])
    rmse5 = rmse(true_x[1:5], pred_x[1:5])
    rmse10 = rmse(true_x[1:10], pred_x[1:10])
    rmse20 = rmse(true_x[1:20], pred_x[1:20])
    # return skipmissing(rmse1), skipmissing(rmse5), skipmissing(rmse10), skipmissing(rmse20)
    return rmse1,rmse5,rmse10,rmse20
end



FILES_ECXEPT = []

function read_ude_c(train_size)
    # Reads UDE predictions and computes RMSE for c and forecasts.
    # train_size: size of the training data used for UDE predictions
    RMSE = []
    for rnd in rnds
        filename = "outputs_latest/ude/ude_predictions_$(train_size)_$(rnd).jld2"
        try
            data = JLD2.load(filename)  # Attempt to load the file
            rmse_c = rmse(data["predictions"].true_c, data["predictions"].pred_c)
            rmse1,rmse5,rmse10,rmse20 = eval_forecast(data["predictions"].true_x,
                                                         data["predictions"].preds[1,:])
            push!(RMSE, Dict("method"=>"UDE","seed"=>parse(Int, rnd),"rmseC"=>rmse_c,
                        "rmse1"=>rmse1,"rmse5"=>rmse5,"rmse10"=>rmse10,"rmse20"=>rmse20))
        catch e
            println("Error loading file '$filename': $e", rnd)
            global FILES_ECXEPT  # Ensure FILES_ECXEPT is accessible
            push!(FILES_ECXEPT, filename)  # Store the filename that caused the error
        end
    end
    return DataFrame(RMSE)
end



function read_gp_c(train_size)
    RMSE = []
    for rnd in rnds
        filename = "outputs_latest/gp/gp_predictions_$(train_size)_$(rnd).jld2"
        try
            data = JLD2.load(filename)  # Attempt to load the file
            rmse_c = rmse(data["predictions"].true_c, data["predictions"].pred_c)
            rmse1,rmse5,rmse10,rmse20 = eval_forecast(data["predictions"].true_x, 
                                                        data["predictions"].preds[1,:])
            push!(RMSE, Dict("method"=>"GP","seed"=>parse(Int, rnd),"rmseC"=>rmse_c,
                "rmse1"=>rmse1,"rmse5"=>rmse5,"rmse10"=>rmse10,"rmse20"=>rmse20))
        catch e
            println("Error loading file '$filename': $e", rnd)
            global FILES_ECXEPT  # Ensure FILES_ECXEPT is accessible
            push!(FILES_ECXEPT, filename)  # Store the filename that caused the error
        end
    end
    return DataFrame(RMSE)
end


function read_kernel_c(train_size)
    RMSE = []
    for rnd in rnds
        filename = "outputs_latest/kernel/kernel_predictions_$(train_size)_$(rnd).jld2"
        try
            data = JLD2.load(filename)  # Attempt to load the file
            rmse_c = rmse(data["predictions"].true_c, data["predictions"].pred_c)
            rmse1,rmse5,rmse10,rmse20 = eval_forecast(data["predictions"].true_x, 
                                                        data["predictions"].preds)
            push!(RMSE, Dict("method"=>"KERNEL","seed"=>parse(Int, rnd),"rmseC"=>rmse_c,
                "rmse1"=>rmse1,"rmse5"=>rmse5,"rmse10"=>rmse10,"rmse20"=>rmse20))
        catch e
            println("Error loading file '$filename': $e", rnd)
            global FILES_ECXEPT  # Ensure FILES_ECXEPT is accessible
            push!(FILES_ECXEPT, filename)  # Store the filename that caused the error
        end
    end
    return DataFrame(RMSE)
end



# Read UDE predictions and compute RMSE for different training sizes
rmseUDE204 = read_ude_c(204)
CSV.write("csv/rmseUDE204.csv", rmseUDE204)
println()

rmseUDE530 = read_ude_c(530)
CSV.write("csv/rmseUDE530.csv", rmseUDE530)
println()

rmseUDE800 = read_ude_c(800)
CSV.write("csv/rmseUDE800.csv", rmseUDE800)

println()

rmseKERNEL204 = read_kernel_c(204)
println(size(rmseKERNEL204))
CSV.write("csv/rmseKERNEL204.csv", rmseKERNEL204)
println()

rmseKERNEL530 = read_kernel_c(530)
println(size(rmseKERNEL530))
CSV.write("csv/rmseKERNEL530.csv", rmseKERNEL530)
println()

rmseKERNEL800 = read_kernel_c(800)
println(size(rmseKERNEL800))
CSV.write("csv/rmseKERNEL800.csv", rmseKERNEL800)


rmseGP204 = read_gp_c(204)
println(size(rmseGP204))
CSV.write("csv/rmseGP204.csv", rmseGP204)
println()

rmseGP530 = read_gp_c(530)
println(size(rmseGP530))
CSV.write("csv/rmseGP530.csv", rmseGP530)
println()

rmseGP800 = read_gp_c(800)
println(size(rmseGP800))
CSV.write("csv/rmseGP800.csv", rmseGP800)


println("Done reading all files.")
println("=============================================")
println("Files that could not be loaded:")
for file in FILES_ECXEPT
    println(file)
end
println("=============================================")



# Define the directory containing CSV files
function get_avg_time(train_size, method)
    pattern = "outputs/$(method)/results_$(train_size)_*.csv"
    csv_files = glob(pattern)
    df_list = [CSV.read(file, DataFrame) for file in csv_files]
    merged_df = vcat(df_list...)
    time_list = [parse(Int, replace(t, " milliseconds"=>""))/1000 for t in merged_df[!,"time"]]
    return mean(time_list)
end


# Function to filter outliers for a given column using IQR
function filter_outliers(data::Vector{Float64})
    q1, q3 = quantile(data, [0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return filter(x -> lower_bound <= x <= upper_bound, data)
end

# Calculate table:
#lets prepare table:
train_sizes = [204,530,800]
methods = [ "KERNEL", "GP", "UDE"]
RMSE_df = DataFrame(training_size=Int[], method=String[], 
            rmse1=Float64[], rmse5=Float64[], rmse10=Float64[], rmse20=Float64[],
            train_time=Float64[])

for t in train_sizes
    for m in methods
        filename = "csv/rmse$(m)$(t).csv"
        df = CSV.read(filename, DataFrame)
        println(filename)
        avg_time = get_avg_time(t, m)
        row = (training_size = t, method= m, 
                rmse1 = median(df.rmse1), rmse5 = median(df.rmse5),
                rmse10 = median(df.rmse10), rmse20 = median(df.rmse20),
                train_time = round(avg_time, digits=0))
        # row = (training_size = t, method= m, 
        #         rmse1 = mean(filter_outliers(df.rmse1)), 
        #         rmse5 = mean(filter_outliers(df.rmse5)),
        #         rmse10 = mean(filter_outliers(df.rmse10)), 
        #         rmse20 = mean(filter_outliers(df.rmse20)),
        #         train_time = round(avg_time, digits=0))
        push!(RMSE_df, row)
    end
end
RMSE_df[!, "rmse1"] .= round.(RMSE_df[!, "rmse1"], digits=3)
RMSE_df[!, "rmse5"] .= round.(RMSE_df[!, "rmse5"], digits=3)
RMSE_df[!, "rmse10"] .= round.(RMSE_df[!, "rmse10"], digits=3)
RMSE_df[!, "rmse20"] .= round.(RMSE_df[!, "rmse20"], digits=3)

CSV.write("csv/rmse_median_table.csv", RMSE_df)

# Perform Statistical test:
# Example RMSE values for 800 datapoints:
# Perform paired t-tests
# println("-----------------------Stats Test-------------------------")
# println("Kernel vs UDE:")
# println(OneSampleTTest(rmseKERNEL800.rmse20 .- rmseUDE800.rmse20))

# println("Kernel vs GP:")
# println(OneSampleTTest(rmseKERNEL800.rmse20 .- rmseGP800.rmse20))

# println("UDE vs GP:")
# println(OneSampleTTest(rmseUDE800.rmse20 .- rmseGP800.rmse20))
println("-------------------------ANOVA---------------------------")
df = vcat(rmseKERNEL800, rmseUDE800, rmseGP800)

# Method 1: HypothesisTests (simpler output)
kernel_data = df[df.method .== "KERNEL", :rmse20]
ude_data = df[df.method .== "UDE", :rmse20]
gp_data = df[df.method .== "GP", :rmse20]

anova_test = OneWayANOVATest(ude_data, gp_data)
println("HypothesisTests result:")
println(anova_test)
println("P-value: ", pvalue(anova_test))

println("---------------------------------------------------------")

# HypothesisTests result:
# One-way analysis of variance (ANOVA) test
# -----------------------------------------
# Population details:
#     parameter of interest:   Means
#     value under h_0:         "all equal"
#     point estimate:          NaN

# Test summary:
#     outcome with 95% confidence: fail to reject h_0
#     p-value:                     0.6558

# Details:
#     number of observations: [50, 50]
#     F statistic:            0.199839
#     degrees of freedom:     (1, 98)

# P-value: 0.6558374723462265







# Read CSV file for one method
# rmseARIMA800 = CSV.read("outputs/arima/results_800.csv", DataFrame)
# rmseLSTM800 = CSV.read("outputs/lstm/results_800.csv", DataFrame)

# df800 = vcat(rmseUDE800, rmseKERNEL800, rmseGP800)
# # Start plotting:
# # Reshape data into a long format
# # Reshape data into a long format
# df_long = stack(df800, [:rmse1, :rmse5, :rmse10, :rmse20]; 
#             variable_name=:rmse_type, value_name=:rmse_value)

# # Generate the grouped boxplot
# @df df_long groupedboxplot(:rmse_type, 
#                 :rmse_value, 
#                 group=:method, 
#                 xlabel="RMSE Types", 
#                 ylabel="RMSE Values", 
#                 legend=:topright,
#                 bar_width=0.4,dpi=300,
#                 yscale=:log10)
# # ylims!(0.0, 20.0)
# savefig("plots/grouped_rmsebox2.png")