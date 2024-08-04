# redraw the c estimates
using Pkg
Pkg.activate("env")

using JLD
using Plots
using ComponentArrays
using LaTeXStrings


@load "results/kernel_predicted_params.jld" params
kp = ComponentArray(params)

@load "results/GP_predicted_params.jld" params
gp = ComponentArray(params)

@load "results/UDE_predicted_params.jld" params solution
nnp = ComponentArray(params)
nnsol = ComponentArray(solution)


c_real = collect(range(0.0, 4.0, length=1001))[2:end]

plot(c_real, label=latexstring("true \$c\$"))
plot!(kp.uhat[2:end, 3], label=latexstring("Kernel function Approx. \$̂c\$"))
plot!(gp.uhat[2:end, 3], label=latexstring("Gaussian Kernel Approx. \$̂c\$"))
pc = plot!(nnsol.uhat[3,:], color="purple", markerstrokewidth =0, marker=(:dot,1),
        label=latexstring("UDE NN method \$̂c\$"), xlabel="Timesteps", 
        ylabel="unresolved variable values")
png(pc, "results/C_hat_compare.png")