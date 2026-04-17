# ============================================================
# CStar Comparison v2: UDE_i | GP_i | ARIMA | RandomWalk | LSTM
# State: [x, c, i] — i jointly inferred (latent)
#
# Speed improvements over v1 (cstar_comparison_i_dip.jl):
#   1. uhat split optimization: NN weights and uhat are updated in
#      separate ADAM steps — halves the AD graph at each step.
#      NN params: Zygote AD (as before)
#      uhat:      direct analytical gradient (no AD through uhat)
#   2. ADAM iterations: 3200 → 1500; L-BFGS g_abstol: 1e-2 → 5e-2
#      + patience-based early stop (200 iters no improvement)
#   3. NN_c shrunk: 1→16→16→1 with tanh (fewer params, cheaper AD)
#      NN_i unchanged (already small)
#   4. Float32 throughout UDE (uhat, NN init, loss arithmetic)
#   5. Dynamical loss: kept stepwise — batching is NOT safe here
#      because step() is sequential (x[t] depends on x[t-1] via
#      physics; unrolling produces a length-T AD chain regardless).
#      Instead the gain comes from items 1 & 4 above.
#
# Plot additions vs v1:
#   - c_hat_plot: y-axis clamped to [-0.5, 10.0] for readability
#   - regime_overlay_plot: 3-panel (c+det shading / x+dip shading /
#     stacked regime bars det vs dip) — mirrors tilman_detector.jl style
#
# Dual regime evaluation (unchanged from v1):
#   det F1:  regime ground truth = Cstar thresholds on c_real
#   dip F1:  regime ground truth = dip-test detector on x trajectory
# ============================================================

using Pkg
Pkg.activate("/home/ceoas/rathorek/projects/CStar/env_c")

using LinearAlgebra, Distributions, Statistics, Random
using Plots
using Zygote
using AbstractGPs, KernelFunctions
import AbstractGPs: mean_vector
using ParameterHandling
using ParameterHandling: flatten
using StableRNGs
using JLD2, CSV, DataFrames, Dates
using Flux: Optimise
using StateSpaceModels
using ComponentArrays, Lux, Optimisers
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LineSearches
using DataStructures: OrderedDict
using PyCall
using Printf


# ── CLI args ──────────────────────────────────────────────────────────────────
# Usage: julia script.jl <train_size> <seed> [consumption]
#   consumption in {holling3, quadratic, linear, constant}  (default: holling3)
@assert length(ARGS) >= 2 "Usage: julia script.jl <train_size> <seed> [consumption]"
train_size = parse(Int64, ARGS[1])
seed       = parse(Int64, ARGS[2])
cons_arg   = length(ARGS) >= 3 ? ARGS[3] : "holling3"

const VALID_CONSUMPTION = ("holling3", "quadratic", "linear", "constant")
@assert cons_arg in VALID_CONSUMPTION "consumption must be one of: $(join(VALID_CONSUMPTION, ", "))"
const CONSUMPTION_TYPE = Symbol(cons_arg)

Random.seed!(seed)
rng = StableRNG(seed)

base_dir = "/home/ceoas/rathorek/projects/CStar/revision_exp/comp_op/obs_x_$(cons_arg)"
plot_dir = "$base_dir/ts$(train_size)_seed$(seed)"
mkpath(plot_dir)

println("="^60)
println("CStar Comparison v2 (i + dip) | Train: $train_size | Seed: $seed")
println("="^60)

# ── Known system parameters ───────────────────────────────────────────────────
const R_PARAM  = 1.0
const K_PARAM  = 10.0
const H_PARAM  = 1.0
const T_PARAM  = 30.0
const HORIZON  = 20
const DATASIZE = 400

const CSTAR1 = 1.7944
const CSTAR2 = 2.5964

const F_TOL = 1e-8

# ── Consumption functional ────────────────────────────────────────────────────
# Controlled via CLI arg 3.  phi(x) in the harvesting term c*phi(x):
#   holling3  ->  x^2/(x^2+h^2)   [Tilman / type-II functional response]
#   quadratic ->  x^2
#   linear    ->  x
#   constant  ->  1               [pure proportional: harvest = c]
consumption(x) =
    CONSUMPTION_TYPE === :holling3  ? x^2 / (x^2 + H_PARAM^2) :
    CONSUMPTION_TYPE === :quadratic ? x^2 :
    CONSUMPTION_TYPE === :linear    ? x   :
    CONSUMPTION_TYPE === :constant  ? one(x) :
    error("Unknown CONSUMPTION_TYPE: $CONSUMPTION_TYPE")
    
# ── Data generation ───────────────────────────────────────────────────────────
function tilman_system(u0, datasize, rng)
    x = zeros(datasize);  i = zeros(datasize)
    x[1], i[1] = u0
    c = range(0.0, 4.0; length=datasize)
    for k in 2:datasize
        η    = rand(rng, Normal(0, 0.07))
        xₖ   = x[k-1];  iₖ = i[k-1];  cₖ = c[k]
        x[k] = xₖ + R_PARAM*xₖ*(1 - xₖ/K_PARAM) - cₖ*consumption(xₖ) + iₖ*xₖ
        i[k] = (1 - 1/T_PARAM)*iₖ + η
    end
    return copy(transpose(cat(x, i, dims=2)))
end

function data_prep(train_size, rng)
    data   = Array(tilman_system(Float32[10.0; 0.0], DATASIZE, rng))
    x̄      = mean(data, dims=2)
    data_n = data .+ (5e-2 * x̄) .* randn(rng, eltype(data), size(data))
    data_n[1, :] .= max.(data_n[1, :], 0.0)
    X_train = data_n[:, 1:train_size]
    X_test  = data_n[1, train_size+1:train_size+HORIZON]
    return X_train, X_test, data_n, data   # data = clean simulation (no obs noise)
end

X_train, X_test, data_n, data_clean = data_prep(train_size, rng)
c_real      = collect(range(0.0, length=DATASIZE, stop=4.0))
x_obs_train = X_train[1, :]

println("Data ready | train=$(size(X_train)) | test=$(length(X_test))")

# ── Metrics ───────────────────────────────────────────────────────────────────
rmse(y, ŷ) = sqrt(mean((y .- ŷ).^2))
mae(y, ŷ)  = mean(abs.(y .- ŷ))
mape(y, ŷ) = mean(abs.((y .- ŷ) ./ (abs.(y) .+ 1e-10))) * 100

# ══════════════════════════════════════════════════════════════════════════════
# DIP-TEST REGIME DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
const DIP_STABLE_HIGH = 0
const DIP_FLICKERING  = 1
const DIP_STABLE_LOW  = 2

const _diptest_ref = Ref{PyObject}()
function _ensure_diptest()
    isassigned(_diptest_ref) || (_diptest_ref[] = pyimport("diptest"))
    return _diptest_ref[]
end

function dip_pvalue(x::Vector{Float64})::Float64
    dt = _ensure_diptest()
    std(x) < 1e-6 && return 1.0
    return Float64(dt.diptest(x)[2])
end

struct DipSignals
    norm_var::Vector{Float64}
    dip_p::Vector{Float64}
    win_mean::Vector{Float64}
end

function dip_detect_signals(x::Vector{Float64}; window::Int=80)
    n    = length(x)
    half = window ÷ 2
    norm_var = fill(NaN, n)
    dip_p    = fill(NaN, n)
    win_mean = fill(NaN, n)
    dx = vcat(0.0, diff(x))
    for t in (half+1):(n-half)
        seg_x  = x[t-half:t+half-1]
        seg_dx = dx[t-half:t+half-1]
        m = mean(seg_x)
        win_mean[t] = m
        if std(seg_dx) > 1e-9
            norm_var[t] = var(seg_dx) / max(m, 1e-3)^2
        end
        dip_p[t] = dip_pvalue(seg_x)
    end
    return DipSignals(norm_var, dip_p, win_mean)
end

function dip_calibrate_threshold(baseline_nv::Vector{Float64}; k::Float64=2.5)
    nv = filter(v -> !isnan(v) && v > 0, baseline_nv)
    length(nv) < 5 && return nothing
    log_nv = log10.(nv)
    return 10.0 ^ (mean(log_nv) + k * std(log_nv))
end

function dip_binary_closing(mask::AbstractVector{Bool}, ksize::Int)
    n    = length(mask)
    half = ksize ÷ 2
    dilated = falses(n)
    for t in 1:n
        a = max(1, t-half); b = min(n, t+half)
        @views any(mask[a:b]) && (dilated[t] = true)
    end
    eroded = falses(n)
    for t in 1:n
        a = max(1, t-half); b = min(n, t+half)
        @views all(dilated[a:b]) && (eroded[t] = true)
    end
    return eroded
end

function dip_label_regimes(x::Vector{Float64}, sig::DipSignals,
                            var_thr::Union{Nothing,Float64};
                            low_guard::Float64=2.0, lookback::Int=20,
                            dip_alpha::Float64=0.05,
                            high_thresh::Float64=4.0, low_thresh::Float64=2.0,
                            closing_size::Int=30)
    n     = length(x)
    valid = .!(isnan.(sig.win_mean) .| isnan.(sig.dip_p))
    flick_dip = (sig.dip_p .< dip_alpha) .& valid
    flick_var = falses(n)
    if var_thr !== nothing
        for t in 1:n
            valid[t] && !isnan(sig.norm_var[t]) && sig.norm_var[t] > var_thr &&
                (flick_var[t] = true)
        end
    end
    flick_closed = dip_binary_closing(flick_dip .| flick_var, closing_size) .& valid
    regime = fill(-1, n)
    for t in 1:n
        !valid[t] && continue
        if t > lookback && all(x[t-lookback:t-1] .< low_guard)
            regime[t] = DIP_STABLE_LOW; continue
        end
        if flick_closed[t]
            regime[t] = DIP_FLICKERING
        elseif sig.win_mean[t] > high_thresh
            regime[t] = DIP_STABLE_HIGH
        elseif sig.win_mean[t] < low_thresh
            regime[t] = DIP_STABLE_LOW
        else
            regime[t] = DIP_FLICKERING
        end
    end
    return regime
end

function run_dip_detector(x_full::Vector{Float64}, c_real_full::Vector{Float64};
                           window::Int=80, closing_size::Int=30)
    sig         = dip_detect_signals(x_full; window=window)
    baseline_nv = sig.norm_var[findall(c -> c < CSTAR1, c_real_full)]
    var_thr     = dip_calibrate_threshold(baseline_nv)
    regime      = dip_label_regimes(x_full, sig, var_thr; closing_size=closing_size)
    return regime, sig, var_thr
end

function print_dip_summary(regime::Vector{Int}, c_real::Vector{Float64}, train_size::Int)
    reg_train = regime[1:train_size]
    valid_idx = findall(r -> r >= 0, reg_train)
    n_high  = count(==(DIP_STABLE_HIGH), reg_train[valid_idx])
    n_flick = count(==(DIP_FLICKERING),  reg_train[valid_idx])
    n_low   = count(==(DIP_STABLE_LOW),  reg_train[valid_idx])
    n_inv   = train_size - length(valid_idx)
    flick_idx = findall(==(DIP_FLICKERING), reg_train)
    fs, fe, density = isempty(flick_idx) ? (-1, -1, 0.0) :
        (flick_idx[1], flick_idx[end],
         count(==(DIP_FLICKERING), reg_train[flick_idx[1]:flick_idx[end]]) /
         length(flick_idx[1]:flick_idx[end]))
    println("\n── Dip-test regime summary (train window 1:$train_size) ─────────────────")
    @printf("  stable_high : %4d | flickering : %4d | stable_low : %4d | invalid : %4d\n",
            n_high, n_flick, n_low, n_inv)
    fs > 0 && @printf("  flick span  : [%d, %d]  density=%.3f\n", fs, fe, density)
    det_cls = [c < CSTAR1 ? "high" : c < CSTAR2 ? "flick" : "low" for c in c_real[1:train_size]]
    println("  ── vs deterministic ──────────────────────────────────────────────")
    @printf("  det  high/flick/low : %d / %d / %d\n",
            count(==("high"),det_cls), count(==("flick"),det_cls), count(==("low"),det_cls))
    println("────────────────────────────────────────────────────────────────────")
end

dip_regime_to_class(r::Int) = r == DIP_STABLE_HIGH ? 1 :
                               r == DIP_FLICKERING  ? 2 :
                               r == DIP_STABLE_LOW  ? 3 : -1

# ── Run detector up front ─────────────────────────────────────────────────────
println("\nRunning dip-test detector on observed x trajectory...")
x_full_obs     = Float64.(data_n[1, :])
dip_regime_all, dip_sig, dip_var_thr = run_dip_detector(x_full_obs, c_real)
dip_regime_train = dip_regime_all[1:train_size]
print_dip_summary(dip_regime_all, c_real, train_size)

# ── Classification helpers ────────────────────────────────────────────────────
categorize_c(c_vec) = [c < CSTAR1 ? 1 : c < CSTAR2 ? 2 : 3 for c in c_vec]

function confusion_matrix_3(true_cls::Vector{Int}, pred_cls::Vector{Int})
    cm = zeros(Int, 3, 3)
    for (t, p) in zip(true_cls, pred_cls); cm[t, p] += 1; end
    return cm
end

function f1_scores(cm)
    tp   = diag(cm)
    fp   = vec(sum(cm, dims=1)) .- tp
    fn   = vec(sum(cm, dims=2)) .- tp
    prec = @. ifelse(tp+fp > 0, tp/(tp+fp), 0.0)
    rec  = @. ifelse(tp+fn > 0, tp/(tp+fn), 0.0)
    f1   = @. ifelse(prec+rec > 0, 2prec*rec/(prec+rec), 0.0)
    w    = vec(sum(cm, dims=2)) ./ sum(cm)
    return f1, mean(f1), dot(f1, w)
end

function classification_metrics_det(true_c, pred_c, label)
    valid = isfinite.(pred_c) .& isfinite.(true_c) .& (pred_c .> 0) .& (true_c .> 0)
    n_valid = sum(valid)
    println("$label [det]: $(n_valid)/$(length(pred_c)) points (c>0) used")
    n_valid < 3 && return Dict("f1_1"=>NaN,"f1_2"=>NaN,"f1_3"=>NaN,
                               "macro_f1"=>NaN,"weighted_f1"=>NaN)
    cm = confusion_matrix_3(categorize_c(true_c[valid]), categorize_c(pred_c[valid]))
    f1, mf1, wf1 = f1_scores(cm)
    println("  CM: $cm\n  F1: $(round.(f1,digits=4)) | Macro: $(round(mf1,digits=4)) | Weighted: $(round(wf1,digits=4))")
    return Dict("f1_1"=>round(f1[1],digits=5),"f1_2"=>round(f1[2],digits=5),
                "f1_3"=>round(f1[3],digits=5),
                "macro_f1"=>round(mf1,digits=5),"weighted_f1"=>round(wf1,digits=5))
end

function classification_metrics_dip(dip_regime::Vector{Int}, pred_c, label)
    n = min(length(dip_regime), length(pred_c))
    valid = [i <= n && dip_regime[i] >= 0 && isfinite(pred_c[i]) && pred_c[i] > 0
             for i in 1:n]
    n_valid = sum(valid)
    println("$label [dip]: $(n_valid)/$(n) points (regime≥0, c>0) used")
    n_valid < 3 && return Dict("f1_1_dip"=>NaN,"f1_2_dip"=>NaN,"f1_3_dip"=>NaN,
                               "macro_f1_dip"=>NaN,"weighted_f1_dip"=>NaN)
    idx      = findall(valid)
    true_cls = [dip_regime_to_class(dip_regime[i]) for i in idx]
    pred_cls = categorize_c(pred_c[idx])
    cm = confusion_matrix_3(true_cls, pred_cls)
    f1, mf1, wf1 = f1_scores(cm)
    println("  CM[dip]: $cm\n  F1[dip]: $(round.(f1,digits=4)) | Macro: $(round(mf1,digits=4)) | Weighted: $(round(wf1,digits=4))")
    return Dict("f1_1_dip"=>round(f1[1],digits=5),"f1_2_dip"=>round(f1[2],digits=5),
                "f1_3_dip"=>round(f1[3],digits=5),
                "macro_f1_dip"=>round(mf1,digits=5),"weighted_f1_dip"=>round(wf1,digits=5))
end

# ── Algebraic c (reference) ───────────────────────────────────────────────────
function algebraic_c_sparse(x_obs::AbstractVector{<:Real}, freq::Int)
    xₜ = Float64.(x_obs[1:end-1])
    xₜ₁ = Float64.(x_obs[2:end])
    # Multi-step residual per unit time
    num   = R_PARAM .* xₜ .* (1 .- xₜ./K_PARAM) .+ (xₜ .- xₜ₁) ./ freq
    denom = consumption.(xₜ)
    c_alg = num ./ denom
    c_alg[(xₜ .< 1e-8) .| (abs.(denom) .< 1e-10)] .= NaN
    return c_alg
end
alg_c = max.(algebraic_c_sparse(x_obs_train, FREQ), 0.0)

# ── Physics steps ─────────────────────────────────────────────────────────────
physics_step_x_full(x, c, i_val) =
    clamp(x + R_PARAM*x*(1 - x/K_PARAM) - c*x^2/(x^2 + H_PARAM^2) + i_val*x, 1e-8, 1e4)
i_ar1_step(i_val, innov) = (1 - 1/T_PARAM) * i_val + innov

# ── GP top-level extensions ───────────────────────────────────────────────────
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::ColVecs) = map(m.f, eachcol(x.X))
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::RowVecs) = map(m.f, eachrow(x.X))

function construct_gp(X_scalar, params)
    kernel = params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ)
    return GP(x -> 0.0, kernel)(RowVecs(X_scalar), params.var_noise)
end
function construct_gp_i(X_scalar, params)
    kernel = params.var_kernel_i * Matern52Kernel() ∘ ScaleTransform(params.λ_i)
    return GP(x -> 0.0, kernel)(RowVecs(X_scalar), params.var_noise_i)
end

# ── Shared helpers ────────────────────────────────────────────────────────────
base_res(method, t_secs) = Dict(
    "method"=>method, "seed"=>seed, "train_size"=>train_size,
    "r"=>R_PARAM, "K"=>K_PARAM, "h"=>H_PARAM,
    "time"=>round(t_secs, digits=3))

function fill_metrics!(res, pred_x, X_test)
    for w in [1,5,10,12,15]
        n = min(w, length(pred_x), length(X_test))
        res["rmse$w"] = round(rmse(X_test[1:n], pred_x[1:n]), digits=5)
        res["mae$w"]  = round(mae(X_test[1:n],  pred_x[1:n]), digits=5)
        res["mape$w"] = round(mape(X_test[1:n], pred_x[1:n]), digits=5)
    end
end

# ── CSV utilities ─────────────────────────────────────────────────────────────
function append_csv(path, df)
    lock_path = path * ".lock"
    for _ in 1:100
        if !isfile(lock_path)
            touch(lock_path)
            try
                isfile(path) ? CSV.write(path, df, append=true) : CSV.write(path, df)
            finally
                rm(lock_path, force=true)
            end
            return
        end
        sleep(0.1)
    end
    @error "Could not acquire CSV lock for $path"
end

function standardize_df(df)
    cols = ["method","seed","train_size","r","K","h","time",
            "rmseC","rmseC_alg","rmseI",
            "f1_1","f1_2","f1_3","macro_f1","weighted_f1",
            "f1_1_dip","f1_2_dip","f1_3_dip","macro_f1_dip","weighted_f1_dip",
            "rmse1","rmse5","rmse10","rmse12","rmse15",
            "mae1","mae5","mae10","mae12","mae15",
            "mape1","mape5","mape10","mape12","mape15",
            "x_forecast","c_predicted","i_predicted"]
    for c in cols
        c in names(df) || (df[!, c] = fill(missing, nrow(df)))
    end
    return select(df, cols)
end

# ── Plot constants ────────────────────────────────────────────────────────────
const METHOD_COLORS  = OrderedDict("UDE_i"=>"#e41a1c","GP_i"=>"#377eb8",
                                   "ARIMA"=>"#4daf4a","RW"=>"#984ea3","LSTM"=>"#ff7f00")
const METHOD_MARKERS = OrderedDict("UDE_i"=>:circle,"GP_i"=>:square,
                                   "ARIMA"=>:diamond,"RW"=>:utriangle,"LSTM"=>:star5)
const PLOT_MARGINS = (left_margin=8Plots.mm, bottom_margin=8Plots.mm,
                      right_margin=4Plots.mm, top_margin=4Plots.mm)

const REGIME_COLORS = [:steelblue, :darkorange, :firebrick]
const REGIME_NAMES  = ["stable_high", "flickering", "stable_low"]
const REGIME_ALPHA  = [0.25, 0.30, 0.25]

# ── Helper: contiguous segments for vspan ────────────────────────────────────
function contiguous_segments(indices::Vector{Int})
    isempty(indices) && return Tuple{Int,Int}[]
    segs = Tuple{Int,Int}[]
    s = indices[1]
    for k in 2:length(indices)
        if indices[k] != indices[k-1]+1
            push!(segs, (s, indices[k-1])); s = indices[k]
        end
    end
    push!(segs, (s, indices[end]))
    return segs
end

# ── Plot utilities ────────────────────────────────────────────────────────────
function zoomed_forecast_plot(x_obs, X_test, forecasts, train_size, tag, out_dir)
    z        = max(1, train_size - 50)
    T_te_ext = train_size:train_size+HORIZON
    p = plot(z:train_size, x_obs[z:train_size],
             label="Observed x (train)", color=:gray, alpha=0.7, lw=2,
             xlabel="Time step", ylabel="Abundance (x)",
             title="X Forecast | $tag",
             legend=:best, size=(1100,500), dpi=200; PLOT_MARGINS...)
    plot!(p, T_te_ext, vcat(x_obs[train_size], X_test),
          label="True x (test)", color=:black, lw=2.5)
    for (lbl, fcast) in forecasts
        plot!(p, T_te_ext, vcat(x_obs[train_size], fcast),
              label=lbl, color=get(METHOD_COLORS,lbl,"#333333"), lw=2, ls=:dash,
              marker=get(METHOD_MARKERS,lbl,:circle), ms=4)
    end
    vline!(p, [train_size], color=:black, ls=:dot, lw=1.5, label=false)
    savefig(p, "$out_dir/X_forecast_$(tag).png")
end

"""
c_hat_plot — c recovery with dip-regime shading.
Y-axis clamped to [-0.5, 10.0] so blown-up latent c values don't
collapse the interesting region to a thin strip at the bottom.
"""
function c_hat_plot(true_c, pred_c_dict, alg_c, dip_regime, train_size, tag, out_dir)
    T_c     = 1:train_size
    T_c_alg = 1:length(alg_c)
    # Clamp y range: allow some headroom above CSTAR2, ignore extreme latent values
    y_lo = -0.5
    y_hi = 10.0

    p = plot(T_c, true_c,
             label="True c", color=:black, lw=2.5,
             xlabel="Time step", ylabel="Net harvesting pressure (c)",
             title="C Recovery + Dip Regimes | $tag",
             legend=:best, size=(1100,500), dpi=200; PLOT_MARGINS...)

    # dip-test regime shading using Shape (vspan! ignores label in GR backend)
    y_bot, y_top = y_lo, y_hi
    for (code, col, alpha) in zip(0:2, REGIME_COLORS, REGIME_ALPHA)
        segs = contiguous_segments(findall(i -> i<=train_size && dip_regime[i]==code, 1:train_size))
        isempty(segs) && continue
        for (k, (s, e)) in enumerate(segs)
            lbl = k == 1 ? "dip: $(REGIME_NAMES[code+1])" : nothing
            plot!(p, Shape([s, e, e, s], [y_bot, y_bot, y_top, y_top]);
                  color=col, alpha=alpha, lw=0, label=lbl)
        end
    end

    hline!(p, [0.0],    color=:steelblue, ls=:dash, lw=1.0, label="c = 0")
    hline!(p, [CSTAR1], color="#e6ab02",  ls=:dot,  lw=1.5,
           label="C*₁=$(round(CSTAR1,digits=3))")
    hline!(p, [CSTAR2], color="#d95f02",  ls=:dot,  lw=1.5,
           label="C*₂=$(round(CSTAR2,digits=3))")
    plot!(p, T_c_alg, alg_c, label="Algebraic c",
          color=:gray, lw=1, ls=:dot, alpha=0.7)
    for (lbl, pc) in pred_c_dict
        isnothing(pc) && continue
        plot!(p, 1:length(pc), pc,
              label=lbl, color=get(METHOD_COLORS,lbl,"#333333"), lw=2, ls=:dash, alpha=0.85)
    end
    ylims!(p, y_lo, y_hi)
    savefig(p, "$out_dir/C_hat_$(tag).png")
end

function i_hat_plot(true_i, pred_i_dict, train_size, tag, out_dir)
    T = 1:train_size
    p = plot(T, true_i, label="True i (latent)", color=:black, lw=2.0,
             xlabel="Time step", ylabel="Environmental noise (i)",
             title="I Recovery | $tag",
             legend=:best, size=(1100,500), dpi=200; PLOT_MARGINS...)
    for (lbl, pi) in pred_i_dict
        isnothing(pi) && continue
        plot!(p, 1:length(pi), pi,
              label=lbl, color=get(METHOD_COLORS,lbl,"#333333"), lw=2, ls=:dash, alpha=0.85)
    end
    savefig(p, "$out_dir/I_hat_$(tag).png")
end

"""
regime_overlay_plot — 3-panel diagnostic plot mirroring tilman_detector.jl:

  Panel 1: c(t) true trajectory + Cstar threshold lines, shaded by
           DETERMINISTIC regime (c_real vs CSTAR thresholds).
  Panel 2: x(t) observed trajectory, shaded by DIP-TEST detector regime.
  Panel 3: Stacked regime bars — det (top row) vs dip (bottom row) —
           using the same color vocabulary. Disagreements stand out
           immediately as color mismatches between rows.

This is the key diagnostic for the dual-F1 claim: if det and dip agree,
c recovery is physically meaningful; disagreements reveal seeds where
the stochastic trajectory hadn't yet committed to the forcing-implied regime.
"""
function regime_overlay_plot(x_obs, c_real_full, dip_regime, train_size, tag, out_dir)
    gr()
    ts = 1:train_size

    # Deterministic regime per timestep
    det_regime = [c < CSTAR1 ? 0 : c < CSTAR2 ? 1 : 2 for c in c_real_full[1:train_size]]

    # ── Panel 1: c(t) + deterministic shading ────────────────────────────────
    c_yhi = min(maximum(c_real_full[1:train_size]) * 1.12, 5.0)
    p1 = plot(ts, c_real_full[1:train_size],
              color=:black, lw=2.0, label="c(t)",
              ylabel="c (harvest pressure)", legend=:topleft,
              ylim=(-0.15, c_yhi), title="Deterministic regime (Cstar thresholds)";
              PLOT_MARGINS...)
    hline!(p1, [CSTAR1]; color="#e6ab02", ls=:dot, lw=1.5,
           label="C*₁=$(round(CSTAR1,digits=3))")
    hline!(p1, [CSTAR2]; color="#d95f02", ls=:dot, lw=1.5,
           label="C*₂=$(round(CSTAR2,digits=3))")
    for (code, col) in zip(0:2, REGIME_COLORS)
        segs = contiguous_segments(findall(==(code), det_regime))
        isempty(segs) && continue
        for (k, (s, e)) in enumerate(segs)
            lbl = k == 1 ? "det: $(REGIME_NAMES[code+1])" : nothing
            plot!(p1, Shape([s, e, e, s], [-0.15, -0.15, c_yhi, c_yhi]);
                  color=col, alpha=0.35, lw=0, label=lbl)
        end
    end
    # redraw c(t) line on top of shading
    plot!(p1, ts, c_real_full[1:train_size]; color=:black, lw=2.0, label=nothing)

    # ── Panel 2: x(t) + dip-test shading ─────────────────────────────────────
    x_yhi = max(maximum(x_obs[1:train_size]) * 1.10, 12.0)
    p2 = plot(ts, x_obs[1:train_size],
              color=:black, lw=1.4, label="x (observed)",
              ylabel="x (abundance)", legend=:topright,
              ylim=(-0.5, x_yhi), title="Stochastic regime (dip-test on x)";
              PLOT_MARGINS...)
    for (code, col) in zip(0:2, REGIME_COLORS)
        segs = contiguous_segments(
            findall(i -> i<=train_size && dip_regime[i]==code, 1:train_size))
        isempty(segs) && continue
        for (k, (s, e)) in enumerate(segs)
            lbl = k == 1 ? "dip: $(REGIME_NAMES[code+1])" : nothing
            plot!(p2, Shape([s, e, e, s], [-0.5, -0.5, x_yhi, x_yhi]);
                  color=col, alpha=0.35, lw=0, label=lbl)
        end
    end
    # redraw x(t) line on top of shading
    plot!(p2, ts, x_obs[1:train_size]; color=:black, lw=1.4, label=nothing)

    # ── Panel 3: stacked regime bar chart det (y=1) vs dip (y=0) ─────────────
    # Each timestep is drawn as a tiny horizontal segment.
    # Color mismatches between the two rows = det/dip disagreements.
    p3 = plot(ylabel="regime source", xlabel="timestep",
              ylim=(-0.6, 1.6), legend=:topright,
              yticks=([0.0, 1.0], ["dip", "det"]),
              title="Regime agreement: det (top) vs dip (bottom)";
              PLOT_MARGINS...)
    for (code, col) in zip(0:2, REGIME_COLORS)
        # deterministic row at y=1
        for t in findall(==(code), det_regime)
            plot!(p3, [t-0.5, t+0.5], [1.0, 1.0]; color=col, lw=7, label="")
        end
        # dip row at y=0
        for t in findall(i -> i<=train_size && dip_regime[i]==code, 1:train_size)
            plot!(p3, [t-0.5, t+0.5], [0.0, 0.0]; color=col, lw=7, label="")
        end
    end
    # legend entries
    for (k, name) in enumerate(REGIME_NAMES)
        plot!(p3, [NaN], [NaN]; color=REGIME_COLORS[k], lw=7, label=name)
    end

    layout = @layout [a{0.28h}; b{0.42h}; c{0.30h}]
    fig = plot(p1, p2, p3;
               layout=layout, size=(1100, 900),
               plot_title="Regime overlay | $tag",
               plot_titlefontsize=11,
               left_margin=10Plots.mm, right_margin=6Plots.mm)
    savefig(fig, "$out_dir/regime_overlay_$(tag).png")
    println("Saved regime_overlay_$(tag).png")
    return fig
end

function rmse_window_plot(results_dict, tag, out_dir)
    ws = [1,5,10,12,15]
    p  = plot(xlabel="Forecast horizon (steps)", ylabel="RMSE",
              title="RMSE by Horizon | $tag",
              legend=:best, size=(1100,500), dpi=200, xticks=ws; PLOT_MARGINS...)
    for (lbl, df) in results_dict
        vals = [coalesce(df[1,Symbol("rmse$w")],NaN) for w in ws]
        any(isfinite,vals) || continue
        plot!(p, ws, Float64.(vals), label=lbl,
              color=get(METHOD_COLORS,lbl,"#333333"), lw=2,
              marker=get(METHOD_MARKERS,lbl,:circle), ms=8, msw=1.5)
    end
    savefig(p, "$out_dir/RMSE_window_$(tag).png")
end

function combined_loss_plot(losses_dict, tag, out_dir)
    p = plot(xlabel="Iteration", ylabel="Loss (shifted, log scale)",
             title="Training Loss | $tag",
             legend=:topright, size=(1100,450), dpi=200, yscale=:log10; PLOT_MARGINS...)
    for (lbl, losses) in losses_dict
        isempty(losses) && continue
        shifted = losses .- minimum(losses) .+ 1.0
        plot!(p, 1:length(shifted), shifted, label=lbl,
              color=get(METHOD_COLORS,lbl,"#333333"), lw=1.5, alpha=0.85)
    end
    savefig(p, "$out_dir/loss_combined_$(tag).png")
end

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1: UDE_i
#
# Monolithic Zygote AD over ComponentArray(NNc, NNi, uhat) — same structure
# as v1, keeping the joint gradient so NN and uhat co-evolve correctly.
#
# Changes vs v1:
#  - Float32 throughout (uhat, NN weights, loss arithmetic)
#  - NN_c shrunk: 1→16→16→1 with tanh  (fewer params, cheaper AD)
#  - NN_i:        1→8→8→1  with tanh
#  - ADAM 3200→3000 iters, lr=0.003
#  - L-BFGS g_abstol 1e-2→5e-2; divergence-only stop (no patience)
# ══════════════════════════════════════════════════════════════════════════════
function run_ude_i(X_train, X_test, rng, dip_regime_train)
    println("\n" * "="^60)
    println("METHOD: UDE_i  (v2 — F32, small NN_c/NN_i, tanh)")
    println("="^60)

    ITERS = 1000
    lr    = 0.003f0

    # NN_c: 1→16→16→1 with tanh
    NN_c = Lux.Chain(
        Lux.Dense(1,  16, tanh),
        Lux.Dense(16, 16, tanh),
        Lux.Dense(16, 1))

    # NN_i: 1→8→8→1 with tanh
    NN_i = Lux.Chain(
        Lux.Dense(1, 8, tanh),
        Lux.Dense(8, 8, tanh),
        Lux.Dense(8, 1))

    NNc_params, st_c = Lux.setup(rng, NN_c)
    NNi_params, st_i = Lux.setup(rng, NN_i)

    # Float32 normalization
    x_mean = Float32(mean(X_train[1, :]))
    x_std  = Float32(std(X_train[1, :]))
    normalize_x(x) = [Float32((x - x_mean) / (x_std + 1f-8))]

    # uhat: 3 × T, Float32
    c_init = Float32.(rand(rng, 1, size(X_train, 2)))
    i_init = zeros(Float32, 1, size(X_train, 2))
    uhat   = vcat(Float32.(X_train[1:1, :]), c_init, i_init)

    params0 = ComponentArray(NNc=NNc_params, NNi=NNi_params, uhat=uhat)

    l2_reg(p) = sum(sum(p[k].weight .^ 2) for k in keys(p))

    function step(u, NNc_p, NNi_p)
        x, c, i_val = u[1], u[2], u[3]
        xn     = normalize_x(x)
        Δc     = NN_c(xn, NNc_p, st_c)[1][1]
        ε_i    = NN_i(xn, NNi_p, st_i)[1][1]
        x_next = Float32(clamp(x + R_PARAM*x*(1 - x/K_PARAM) -
                               c*x^2/(x^2 + H_PARAM^2) + i_val*x, 1f-8, 1f4))
        c_next = c + Δc
        i_next = Float32(1 - 1/T_PARAM) * i_val + ε_i
        return [x_next, c_next, i_next]
    end

    x_obs_f32 = Float32.(x_obs_train)

    function loss(params, data)
        𝐮 = params.uhat
        L_dyn = sum(sum((𝐮[:,t] .- step(𝐮[:,t-1], params.NNc, params.NNi)).^2)
                    for t in 2:size(data,2))
        L_obs   = sum((Float32.(data[1:1,:]) .- 𝐮[1:1,:]).^2)
        L_i_reg = sum(𝐮[3,:].^2)
        total = (0.30f0*L_dyn + 0.40f0*L_obs +
                 0.20f0*l2_reg(params.NNc) + 0.05f0*l2_reg(params.NNi) +
                 0.05f0*L_i_reg)
        return isfinite(total) ? total : 1f10
    end

    t_start = now()
    losses  = Float64[]
    callback(state, l) = begin
        push!(losses, l); n = length(losses)
        n % 500 == 0 && println("  UDE_i iter=$n | loss=$(round(l,digits=5))")
        stop = !isfinite(l) || l > 1f8
        stop && println("  UDE_i divergence stop at iter $n")
        return stop
    end

    adtype = Optimization.AutoZygote()
    optf   = Optimization.OptimizationFunction((p, _) -> loss(p, X_train), adtype)
    prob1  = Optimization.OptimizationProblem(optf, params0)
    sol1   = Optimization.solve(prob1, Optimisers.ADAM(lr), callback=callback, maxiters=ITERS)
    prob2  = Optimization.OptimizationProblem(optf, sol1.u)
    lbfgs_losses = Float64[]
    cb2(state, l) = begin
        push!(lbfgs_losses, l); n = length(lbfgs_losses)
        n % 100 == 0 && println("  L-BFGS iter=$n | loss=$(round(l,digits=5))")
        stop = !isfinite(l) || l > 1f8 || (n > 1 && abs(lbfgs_losses[end-1] - l) < F_TOL)
        stop && println("  L-BFGS stop at iter $n")
        return stop
    end
    sol2 = Optimization.solve(prob2, LBFGS(linesearch=BackTracking()),
                              callback=cb2, g_abstol=5f-4)
    append!(losses, lbfgs_losses)
    t_end = now()
    println("UDE_i done in $(canonicalize(t_end-t_start)) | $(length(losses)) total iters | loss=$(round(losses[end],digits=5))")

    # ── Forecast ──────────────────────────────────────────────────────────────
    preds      = fill(NaN32, 3, HORIZON+1)
    preds[:, 1] = [Float32(X_train[1, end]), sol2.u.uhat[2, end], sol2.u.uhat[3, end]]
    for t in 2:(HORIZON+1)
        preds[:, t] = step(preds[:, t-1], sol2.u.NNc, sol2.u.NNi)
    end
    pred_x = Float64.(preds[1, 2:end])
    for k in eachindex(pred_x)
        isfinite(pred_x[k]) || (pred_x[k] = k > 1 ? pred_x[k-1] : X_train[1,end])
    end

    pred_c  = Float64.(sol2.u.uhat[2, :])
    pred_i  = Float64.(sol2.u.uhat[3, :])
    true_c  = c_real[1:train_size]
    true_i  = data_clean[2, 1:train_size]
    alg_idx = findall(isfinite, alg_c)
    c_rmse   = round(rmse(true_c, pred_c),                   digits=5)
    alg_rmse = round(rmse(true_c[alg_idx], alg_c[alg_idx]),  digits=5)
    i_rmse   = round(rmse(true_i, pred_i),                   digits=5)
    println("UDE_i C-RMSE: $c_rmse | Alg C-RMSE: $alg_rmse | I-RMSE: $i_rmse")

    det_metrics = classification_metrics_det(true_c, pred_c, "UDE_i")
    dip_metrics = classification_metrics_dip(dip_regime_train, pred_c, "UDE_i")

    res = merge!(base_res("UDE_i", Dates.value(t_end-t_start)/1000),
                 det_metrics, dip_metrics,
                 Dict("rmseC"=>c_rmse, "rmseC_alg"=>alg_rmse, "rmseI"=>i_rmse,
                      "x_forecast" =>string(round.(pred_x, digits=5)),
                      "c_predicted"=>string(round.(pred_c, digits=5)),
                      "i_predicted"=>string(round.(pred_i, digits=5))))
    fill_metrics!(res, pred_x, X_test)
    return DataFrame(res), pred_x, pred_c, pred_i, losses
end

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2: GP_i  (unchanged from v1 — GP is already fast)
# ══════════════════════════════════════════════════════════════════════════════
function run_gp_i(X_train, X_test, rng, dip_regime_train)
    println("\n" * "="^60)
    println("METHOD: GP_i  (joint c + i inference)")
    println("="^60)

    MAX_ITERS = 2200
    LR        = 0.03

    x_mean = mean(X_train[1, :])
    x_std  = std(X_train[1, :])
    normalize_x(x) = (x - x_mean) / (x_std + 1e-8)

    σ²_k = max(var(x_obs_train), 1e-4)

    flat_initial_params, unflatten = flatten((
        var_kernel   = positive(σ²_k),
        λ            = positive(0.5),
        var_noise    = positive(0.1σ²_k),
        var_kernel_i = positive(0.01σ²_k),
        λ_i          = positive(0.5),
        var_noise_i  = positive(0.001σ²_k),
        uhat = vcat(
            x_obs_train',
            rand(rng, 1, train_size),
            zeros(1, train_size),
        ),
    ))
    unpack = ParameterHandling.value ∘ unflatten

    function objective(params)
        𝐱 = params.uhat[1,:]
        𝐜 = params.uhat[2,:]
        𝐢 = params.uhat[3,:]
        X_norm  = reshape(normalize_x.(𝐱[1:end-1]), :, 1)
        lml_c   = -logpdf(construct_gp(X_norm, params), diff(𝐜))
        i_innov = [𝐢[t] - (1 - 1/T_PARAM)*𝐢[t-1] for t in 2:train_size]
        lml_i   = -logpdf(construct_gp_i(X_norm, params), i_innov)
        L_dyn   = sum((𝐱[t] - physics_step_x_full(𝐱[t-1], 𝐜[t-1], 𝐢[t-1]))^2
                      for t in 2:train_size)
        L_obs   = sum((x_obs_train .- 𝐱).^2)
        L_i_reg = 0.05 * sum(𝐢.^2)
        return 0.35*L_dyn + 0.35*L_obs + lml_c + lml_i + L_i_reg
    end

    opt         = Optimise.ADAM(LR)
    flat_params = deepcopy(flat_initial_params)
    losses      = Float64[]
    t_start     = now()
    for epoch in 1:MAX_ITERS
        if epoch % 500 == 0
            opt.eta /= 2.0
        end        
        grads      = Zygote.gradient(θ -> objective(unpack(θ)), flat_params)
        Optimise.update!(opt, flat_params, grads[1])
        loss_value = objective(unpack(flat_params))
        push!(losses, loss_value)
        epoch % 500 == 0 &&
            println("  GP_i iter=$epoch | loss=$(round(loss_value,digits=5)) | gnorm=$(round(norm(grads),digits=5)) | LR to $(opt.eta)")
        length(losses) > 1 && abs(losses[end-1] - loss_value) < F_TOL &&
            (println("  GP_i early stop at iter $epoch"); break)
    end
    t_end = now()
    println("GP_i done in $(canonicalize(t_end-t_start))")

    final_params = unpack(flat_params)
    c_lat_final  = final_params.uhat[2, :]
    i_lat_final  = final_params.uhat[3, :]
    X_gp_final   = reshape(normalize_x.(final_params.uhat[1, 1:end-1]), :, 1)

    posterior_c = posterior(construct_gp(X_gp_final, final_params), diff(c_lat_final))
    predict_Δc(x) = mean(marginals(posterior_c(RowVecs(reshape([normalize_x(x)],1,1))))[1])

    i_innov_final = [i_lat_final[t] - (1 - 1/T_PARAM)*i_lat_final[t-1]
                     for t in 2:train_size]
    posterior_i  = posterior(construct_gp_i(X_gp_final, final_params), i_innov_final)
    predict_ε_i(x) = mean(marginals(posterior_i(RowVecs(reshape([normalize_x(x)],1,1))))[1])

    preds      = fill(NaN, 3, HORIZON+1)
    preds[:, 1] = [X_train[1, end], c_lat_final[end], i_lat_final[end]]
    for t in 2:(HORIZON+1)
        x_prev, c_prev, i_prev = preds[1,t-1], preds[2,t-1], preds[3,t-1]
        preds[:, t] = [physics_step_x_full(x_prev, c_prev, i_prev),
                       c_prev + predict_Δc(x_prev),
                       i_ar1_step(i_prev, predict_ε_i(x_prev))]
    end
    pred_x = preds[1, 2:end]
    for k in eachindex(pred_x)
        isfinite(pred_x[k]) || (pred_x[k] = k > 1 ? pred_x[k-1] : X_train[1,end])
    end

    true_c   = c_real[1:train_size]
    true_i   = data_clean[2, 1:train_size]
    alg_idx  = findall(isfinite, alg_c)
    c_rmse   = round(rmse(true_c, c_lat_final),  digits=5)
    alg_rmse = round(rmse(true_c[alg_idx], alg_c[alg_idx]), digits=5)
    i_rmse   = round(rmse(true_i, i_lat_final),  digits=5)
    println("GP_i C-RMSE: $c_rmse | Alg C-RMSE: $alg_rmse | I-RMSE: $i_rmse")

    det_metrics = classification_metrics_det(true_c, c_lat_final, "GP_i")
    dip_metrics = classification_metrics_dip(dip_regime_train, c_lat_final, "GP_i")

    res = merge!(base_res("GP_i", Dates.value(t_end-t_start)/1000),
                 det_metrics, dip_metrics,
                 Dict("rmseC"=>c_rmse, "rmseC_alg"=>alg_rmse, "rmseI"=>i_rmse,
                      "x_forecast" =>string(round.(pred_x,      digits=5)),
                      "c_predicted"=>string(round.(c_lat_final, digits=5)),
                      "i_predicted"=>string(round.(i_lat_final, digits=5))))
    fill_metrics!(res, pred_x, X_test)
    return DataFrame(res), pred_x, c_lat_final, i_lat_final, losses
end

# ══════════════════════════════════════════════════════════════════════════════
# METHODS 3 & 4: ARIMA and RandomWalk
# ══════════════════════════════════════════════════════════════════════════════
function run_baseline(name, x_train, X_test, build_model)
    println("\n" * "="^60 * "\nMETHOD: $name\n" * "="^60)
    t0  = now()
    res = merge!(base_res(name, 0.0),
                 Dict("rmseC"=>missing, "rmseC_alg"=>missing, "rmseI"=>missing,
                      "c_predicted"=>missing, "i_predicted"=>missing,
                      "f1_1"=>missing, "f1_2"=>missing, "f1_3"=>missing,
                      "macro_f1"=>missing, "weighted_f1"=>missing,
                      "f1_1_dip"=>missing, "f1_2_dip"=>missing, "f1_3_dip"=>missing,
                      "macro_f1_dip"=>missing, "weighted_f1_dip"=>missing))
    try
        model  = build_model(x_train)
        StateSpaceModels.fit!(model)
        pred_x = reduce(vcat, forecast(model, HORIZON).expected_value)
        res["time"]       = round(Dates.value(now()-t0)/1000, digits=3)
        res["x_forecast"] = string(round.(pred_x, digits=5))
        fill_metrics!(res, pred_x, X_test)
        return DataFrame(res), pred_x
    catch e
        @error "$name failed" exception=e
        res["time"] = missing; res["x_forecast"] = "[]"
        for w in [1,5,10,12,15]; res["rmse$w"]=NaN; res["mae$w"]=NaN; res["mape$w"]=NaN; end
        return DataFrame(res), fill(NaN, HORIZON)
    end
end
run_arima(x_train, X_test)      = run_baseline("ARIMA", x_train, X_test,
    x -> StateSpaceModels.auto_arima(x))
run_randomwalk(x_train, X_test) = run_baseline("RW", x_train, X_test,
    x -> StateSpaceModels.UnobservedComponents(x; trend="random walk"))

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 5: LSTM
# ══════════════════════════════════════════════════════════════════════════════
function run_lstm(X_train, X_test, rng)
    println("\n" * "="^60)
    println("METHOD: LSTM")
    println("="^60)

    SEQ_LEN = 10
    ITERS   = 1500
    LR_LSTM = 0.003f0

    x_train  = X_train[1, :]
    d_mean   = mean(x_train);  d_std = std(x_train)
    norm_x(x)   = (x .- d_mean) ./ d_std
    denorm_x(x) = x .* d_std .+ d_mean
    x_norm = norm_x(x_train)

    function make_sequences(data, L)
        n = length(data) - L
        X = stack([reshape(data[i:i+L-1], 1, L) for i in 1:n]; dims=3)
        y = reshape(data[L+1:end], 1, n)
        return Float32.(X), Float32.(y)
    end
    X_all, y_all = make_sequences(Float32.(x_norm), SEQ_LEN)
    n_val = max(1, Int(floor(0.2*size(X_all,3))))
    n_tr  = size(X_all,3) - n_val
    X_tr = X_all[:,:,1:n_tr];      y_tr = y_all[:,1:n_tr]
    X_va = X_all[:,:,n_tr+1:end];  y_va = y_all[:,n_tr+1:end]

    model_lstm = Lux.Chain(
        Lux.Recurrence(Lux.LSTMCell(1 => 64)),
        Lux.Dense(64=>64, gelu), Lux.Dense(64=>32, gelu),
        Lux.Dense(32=>16, gelu), Lux.Dense(16=>1))
    ps_nt, st_lstm = Lux.setup(rng, model_lstm)
    ps = ComponentArray(ps_nt)

    loss_fn(ps, X, y) = mean((y .- model_lstm(X, NamedTuple(ps), st_lstm)[1]).^2)
    opt_state = Optimisers.setup(Optimisers.Adam(LR_LSTM), ps)
    cur_ps = deepcopy(ps);  best_ps = deepcopy(ps);  best_val = Inf
    losses = Float64[]

    t_start = now()
    for iter in 1:ITERS
        loss_val, grads = Zygote.withgradient(p -> loss_fn(p, X_tr, y_tr), cur_ps)
        push!(losses, loss_val)
        opt_state, cur_ps = Optimisers.update(opt_state, cur_ps, grads[1])
        if iter % 100 == 0
            val_loss = loss_fn(cur_ps, X_va, y_va)
            if val_loss < best_val; best_val = val_loss; best_ps = deepcopy(cur_ps); end
        end
        iter % 500 == 0 &&
            println("  LSTM iter=$iter | loss=$(round(loss_val,digits=6)) | gnorm=$(round(norm(grads[1]),digits=6))")
        length(losses) > 1 && abs(losses[end-1] - loss_val) < F_TOL &&
            (println("  LSTM early stop at iter $iter"); break)
    end
    t_end    = now()
    final_ps = best_val < Inf ? best_ps : cur_ps
    println("LSTM done in $(canonicalize(t_end-t_start))")

    seq = Float32.(x_norm[end-SEQ_LEN+1:end])
    ps_nt_f = NamedTuple(final_ps)
    pred_norm = Float32[]
    for _ in 1:HORIZON
        inp = reshape(seq, (1, SEQ_LEN, 1))
        out, _ = model_lstm(inp, ps_nt_f, st_lstm)
        next_val = out[1,1]
        push!(pred_norm, next_val)
        seq = vcat(seq[2:end], next_val)
    end
    pred_x = Float64.(denorm_x(pred_norm))

    res = merge!(base_res("LSTM", Dates.value(t_end-t_start)/1000),
                 Dict("rmseC"=>missing,"rmseC_alg"=>missing,"rmseI"=>missing,
                      "c_predicted"=>missing,"i_predicted"=>missing,
                      "f1_1"=>missing,"f1_2"=>missing,"f1_3"=>missing,
                      "macro_f1"=>missing,"weighted_f1"=>missing,
                      "f1_1_dip"=>missing,"f1_2_dip"=>missing,"f1_3_dip"=>missing,
                      "macro_f1_dip"=>missing,"weighted_f1_dip"=>missing,
                      "x_forecast"=>string(round.(pred_x, digits=5))))
    fill_metrics!(res, pred_x, X_test)
    return DataFrame(res), pred_x, losses
end

# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL METHODS
# ══════════════════════════════════════════════════════════════════════════════
ude_df, ude_pred_x, ude_pred_c, ude_pred_i, ude_losses = run_ude_i(X_train, X_test, rng, dip_regime_train)
gp_df,  gp_pred_x,  gp_pred_c,  gp_pred_i,  gp_losses  = run_gp_i(X_train, X_test, rng, dip_regime_train)
arima_df, arima_pred_x                                   = run_arima(x_obs_train, X_test)
rw_df,   rw_pred_x                                       = run_randomwalk(x_obs_train, X_test)
lstm_df, lstm_pred_x, lstm_losses                        = run_lstm(X_train, X_test, rng)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
all_dfs = [ude_df, gp_df, arima_df, rw_df, lstm_df]
std_dfs = [standardize_df(df) for df in all_dfs]

tag      = "train$(train_size)_seed$(seed)"
csv_path = "$base_dir/results_ts$(train_size).csv"
for df in std_dfs
    println(df)
    append_csv(csv_path, df)
end
println("\nAll results saved → $csv_path")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
println("\n" * "="^60)
println("CLASSIFICATION SUMMARY  |  $tag")
println("="^60)
@printf("%-10s  %10s %10s  |  %10s %10s\n",
        "Method","macro_f1","wt_f1","macro_dip","wt_dip")
println("-"^60)
for (lbl, df) in zip(["UDE_i","GP_i","ARIMA","RW","LSTM"], std_dfs)
    mf1 = coalesce(df[1,:macro_f1],        NaN)
    wf1 = coalesce(df[1,:weighted_f1],     NaN)
    md  = coalesce(df[1,:macro_f1_dip],    NaN)
    wd  = coalesce(df[1,:weighted_f1_dip], NaN)
    @printf("%-10s  %10.4f %10.4f  |  %10.4f %10.4f\n", lbl,
            isnan(mf1) ? -1.0 : mf1, isnan(wf1) ? -1.0 : wf1,
            isnan(md)  ? -1.0 : md,  isnan(wd)  ? -1.0 : wd)
end
println("="^60)
println("(det=Cstar thresholds on c_real | dip=detector on x trajectory)")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
true_i_train = data_clean[2, 1:train_size]   # clean latent i, no obs noise
gr()

zoomed_forecast_plot(x_obs_train, X_test,
    OrderedDict("UDE_i"=>ude_pred_x,"GP_i"=>gp_pred_x,
                "ARIMA"=>arima_pred_x,"RW"=>rw_pred_x,"LSTM"=>lstm_pred_x),
    train_size, tag, plot_dir)

c_hat_plot(c_real[1:train_size],
    OrderedDict("UDE_i"=>ude_pred_c,"GP_i"=>gp_pred_c),
    alg_c, dip_regime_train, train_size, tag, plot_dir)

i_hat_plot(true_i_train,
    OrderedDict("UDE_i"=>ude_pred_i,"GP_i"=>gp_pred_i),
    train_size, tag, plot_dir)

# NEW: 3-panel regime overlay (det vs dip)
regime_overlay_plot(x_obs_train, c_real, dip_regime_train, train_size, tag, plot_dir)

rmse_window_plot(
    OrderedDict("UDE_i"=>std_dfs[1],"GP_i"=>std_dfs[2],
                "ARIMA"=>std_dfs[3],"RW"=>std_dfs[4],"LSTM"=>std_dfs[5]),
    tag, plot_dir)

combined_loss_plot(
    OrderedDict("UDE_i"=>ude_losses,"GP_i"=>gp_losses,"LSTM"=>lstm_losses),
    tag, plot_dir)

println("\nAll plots saved for $tag")
println("Script completed successfully!")