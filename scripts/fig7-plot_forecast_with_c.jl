# redraw the c estimates
using Pkg
Pkg.activate("env")

using JLD2
using Plots
using ComponentArrays
using LaTeXStrings
using Lux, Zygote
using SciMLBase
using StableRNGs, Distributions

filename = "../outputs/ude/ude_optimal_params_800_86366.jld2"
opm = JLD2.load(filename)
nnsol = opm["solution"]


rng = StableRNG(86366)
#Data generation
u0 = Float32[10.0; 0.0]
datasize = 1001
c_real = collect(range(0.0, length=datasize, stop=4.0))

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

X = Array(tilman(u0,datasize))
data = zeros(size(X))
data[1,:] .= X[1,:] #./ (log(X[1, argmax(X[1,:])])- log(X[1,argmin(X[1,:])]))# X  #log_normalized
data[2,:] .= X[2,:] ./ (X[2, argmax(X[2,:])]- X[2,argmin(X[2,:])])   #normalized i noise  is very much important here.

#Plots.plot(transpose(data), labels=["X-state" "Noise"],  xlabel="Timesteps", ylabel="Abundance")# "C"])
x̄ = mean(X, dims=2)
noise_magnitude = 5e-2
data_n = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))
data_n[1, :] .= max.(data_n[1,:], 0.0)


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

function logistic_growth(u, parameters, states)
    u_hat = NN(u[1:2], parameters.NNparams, states.st)[1] # Network prediction
    x_next = states.r*u[1]*(1-u[1]/states.K)-u[3]*u[1]^2/(u[1]^2+states.h^2)+(1+u[2]) * u[1]       #should be states.c*u[1]^2
    u = [x_next, u[2]+u_hat[1], relu(u[3]+u_hat[2])]     #enforced positive C
end


rectangle(w, h, x, y) = Plots.Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

# using last x value forecast for future:
function forecast_ts(u0, c)
    preds = zeros(3,21)
    preds[:, 1] = u0
    preds[3, 1] = c
    for t in 2:(21)
        preds[:, t] = logistic_growth(preds[:, t-1], nnsol, states) 
        preds[3, t] = c
    end
    return preds[1,:]
end

# Generate Plot
# update as per 204, 530, 800 criterias.

# Set plot attributes
gr(markerstrokewidth=0, ms=4, lw=3,
   xtickfontsize=14, ytickfontsize=14,
   xguidefontsize=16, yguidefontsize=16, 
   legendfontsize=14, dpi=800, margin=6Plots.mm)

plot_array = Any[]

# Panel 1: Environmental State (Includes all legends)
plot(X[1, :], lw=3, ylabel="Environmental state", xlabel="Time", label=L"$y_t$")

rect_positions = [204, 530, 800]
rect_labels = ["B", "C", "D"]

for (i, x) in enumerate(rect_positions)
    plot!(rectangle(20, 17, x, 0), ls=:dash, fillcolor=nothing, label="")
    annotate!(x+10, 1, text(rect_labels[i], 16, :black))  # Bottom mid inside rectangle
end

# Collect all forecast legends in the first plot
pred_labels = [L"$f(c=1.0)$", L"$f(c=2.0)$", L"$f(c=3.0)$", L"$f(c=3.5)$"]
line_styles = [:solid, :dash, :dot, :dashdot]
colors = [:gray40, :gray50, :gray60, :gray70]
marker_styles = [:circle, :square, :diamond, :star]

for (i, label) in enumerate(pred_labels)
    plot!([], [], lw=3, ls=line_styles[i], color=colors[i], marker=(marker_styles[i], 1.6), 
        label=label, foreground_color_legend=nothing, legend_position=(960,16))  # Dummy plots for legend
end

p1 = plot!(legend=:topright)  # Only this subplot has the legend
push!(plot_array, p1)

# Function to generate forecast panels (No Legend)
function forecast_panel(u, start_idx, end_idx, show_ylabel)
    pred_values = [forecast_ts(u, c) for c in [1.0, 2.0, 3.0, 3.5]]

    plot(start_idx-10:start_idx, nnsol.uhat[1, start_idx-10:start_idx], lw=3, label="")
    for (i, pred) in enumerate(pred_values)
        plot!(start_idx:end_idx, pred, marker=(marker_styles[i], 4), lw=3, ls=line_styles[i], 
              color=colors[i], label="")
    end
    return plot!(xlabel="Time", ylabel=show_ylabel ? "Environmental state" : "") 
                      # Y-axis label only for plot 1 and 2
end

# Panel 2: Regime I (Keep y-axis label)
push!(plot_array, forecast_panel(nnsol.uhat[:, 204], 204, 224, true))

# Panel 3: Regime II (No y-axis label)
push!(plot_array, forecast_panel(nnsol.uhat[:, 530], 530, 550, false))

# Panel 4: Regime III (No y-axis label)
push!(plot_array, forecast_panel(nnsol.uhat[:, 800], 800, 820, false))

# Layout and final plot with wider panels
lyt = @layout [a; [b c d]]  # Wider second row
allp = plot(plot_array..., layout=lyt, size=(1600, 875), margin=6Plots.mm,
            titleloc=:left, titlefontsize=18, grid=false)

annotate!(allp[1], -70, 17, text("A", 22, "Helvetica Bold"))
annotate!(allp[2], 189, 11.3, text("B", 22, "Helvetica Bold"))
annotate!(allp[3], 517, 14.5, text("C", 22, "Helvetica Bold"))
annotate!(allp[4], 787, 9.8, text("D", 22, "Helvetica Bold"))
# Save plot
png(allp, "../plots/forecast_2x3.png")
