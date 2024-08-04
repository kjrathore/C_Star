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
using JLD

rng = StableRNG(42)


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
# size(X), size(y)


states = (r=1.0, K=10.0, h=1.0, T=30.0)

# Declare model parameters using `ParameterHandling.jl` types.
flat_initial_params, unflatten = flatten((
    var_kernel = positive(0.6),
    λ = positive(2.5),
    var_noise = positive(0.1),
    uhat = hcat(data_n, zeros(size(data_n)[1],1)),
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


# workaround needed for Zygote
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::ColVecs) = map(m.f, eachcol(x.X))
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::RowVecs) = map(m.f, eachrow(x.X))

# zygote = Zygote.gradient(n -> model(n; mean=x->0.), 1.)

function relu(x)
    return max.(x, 0)
end

# Specify an objective function in terms of x and y.
function objective(params)
    gp = construct_finite_gp(X, params)

    function infer(xtest, gp)
        post = posterior(gp, y)
        # println("created post")
        pred_y = marginals(post(RowVecs(xtest)))
        # println("got pred_y")
        μ = mean.(pred_y)
        return μ
    end
    
    function logistic_growth(u)
        # this will predict two dimensional array. [di]
        ut = u[1:1000, :]
        di = infer(ut[:, 1:2], gp)    #will predict entire y vector
        # println("di:",size(di))
        x_next = states.r .* ut[:,1] .* (1 .-ut[:,1] ./ states.K) .- ut[:,3] .* (ut[:,1].^2 ./ (ut[:,1].^2 .+ states.h .^2)) .+ (1 .+ ut[:,2]) .* ut[:,1]  
        # println("x_next: ", x_next)   
        u_next = hcat(x_next, ut[:,2].+di, relu(ut[:,3]))     #enforced positive C
        return u_next
    end
    
    function loss(params)
        # println("uhat", size(uhat))
        bar_ut = logistic_growth(params.uhat)   #we will infer at once
        # println("barut", size(bar_ut))
        # skip x0, initial values and compare
        L_dyn = sum((params.uhat[2:end,:] .- bar_ut).^2)
        L_obs = sum((X .- params.uhat[1:1000,1:2]).^2)                     #observational Loss
    
        lml = -logpdf(gp, y)
        # print(L_dyn + L_obs + lml)
        return L_dyn + L_obs + lml
    end 
    return loss(params)
end

println("--------------Started Optimizing----------------------------")

training_results = Optim.optimize(
    objective ∘ unpack,
    θ -> only(Zygote.gradient(objective ∘ unpack, θ)),
    flat_initial_params, # Add some noise to make learning non-trivial
    BFGS(
        alphaguess = Optim.LineSearches.InitialStatic(scaled=true),
        linesearch = Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(show_trace = true);
                    # iterations = 2);
    inplace=false,
)

println("--------------Completed Optimizing----------------------------")

final_params = unpack(training_results.minimizer)

#define mean sequared deviation
msd(a, b) = mean(abs2, a-b)

c_real = collect(range(0.0, 4.0, length=1001))[2:end]
c_hat = final_params.uhat[2:end,3]
println("Mean Absolute test error: ",mean(abs.(c_real - c_hat)))
println("Root mean square deviation RMSD: ", sqrt(msd(c_real, c_hat)), "\n")

out_dir = "$(pwd())/model_evals/results/GP"

plot(c_real, label=latexstring("true \$c\$"))
pc = plot!(c_hat, label=latexstring("predicted \$̂c\$"), xlabel="Timesteps", ylabel="unresolved variable values")

png(pc, "$(out_dir)/GP_C_estimation.png")


#Saving trained model parameters:
params = NamedTuple(final_params)
# solution = NamedTuple(sol.u)
@save "./results/GP_predicted_params.jld" params 

# * time: 1283.9049999713898
# 716    -1.504544e+03     1.858386e-02
# * time: 1287.7009999752045
# --------------Completed Optimizing----------------------------
# Mean Absolute test error: 0.28420247032846957
# Root mean square deviation RMSD: 0.36380387296710465
