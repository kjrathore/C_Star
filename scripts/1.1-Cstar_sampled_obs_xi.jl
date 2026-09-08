# ============================================================
# CStar Comparison v3-sparse: UDE_obs | GP_obs | ARIMA | RW | LSTM
# State visible to UDE/GP: [x, i]  (both observed, with noise)
# Latent:                  c only  (unobserved, to be recovered)
#
# Sparse-sampling extension of cstar_comparison_obs_xi.jl:
#   SAMPLE_FREQ (CLI arg 4, default 1) controls observation frequency.
#   SAMPLE_FREQ=3 means only t=1,4,7,10,... are observed from the
#   dense simulation — i.e. every 3rd timestep.
#
# Design choices for the sparse case:
#   - Simulation always runs at full dense resolution (DATASIZE steps).
#   - Subsampling picks obs_idx = 1:SAMPLE_FREQ:train_size from noisy data.
#   - uhat lives at observed indices only (N_obs columns, not train_size).
#   - Dynamical loss bridges consecutive observed pairs by rolling physics
#     SAMPLE_FREQ steps (multi-step forward map, no continuous ODE needed).
#   - algebraic_c is adapted: diff over SAMPLE_FREQ steps in numerator.
#   - ARIMA / RW / LSTM receive the sparse x series (N_obs points).
#   - Dip detector runs on the full dense x trajectory (available from sim).
#   - c_real and regime labels evaluated at observed indices for F1/RMSE.
#   - Forecast (HORIZON steps) is always dense, step-by-step from last obs.
#
# Dual regime evaluation (identical to v3 dense):
#   det F1:  Cstar thresholds on c_real at observed indices
#   dip F1:  dip-test detector labels at observed indices
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
# Usage: julia script.jl <train_size> <seed> [consumption] [sample_freq]
#   consumption in {holling3, quadratic, linear, constant}  (default: holling3)
#   sample_freq: positive integer, every Nth timestep observed (default: 1)
@assert length(ARGS) >= 2 "Usage: julia 1.1-Cstar_sampled_obs_xi.jl <train_size> <seed> [consumption] [sample_freq]"
# julia 1.1-Cstar_sampled_obs_xi.jl 380 42 holling3 2

train_size  = parse(Int64, ARGS[1])
seed        = parse(Int64, ARGS[2])
cons_arg    = length(ARGS) >= 3 ? ARGS[3] : "holling3"
sample_freq = length(ARGS) >= 4 ? parse(Int64, ARGS[4]) : 3

const VALID_CONSUMPTION = ("holling3", "quadratic", "linear", "constant")
@assert cons_arg in VALID_CONSUMPTION "consumption must be one of: $(join(VALID_CONSUMPTION, ", "))"
@assert sample_freq >= 1 "sample_freq must be >= 1"
const CONSUMPTION_TYPE = Symbol(cons_arg)
const SAMPLE_FREQ      = sample_freq

Random.seed!(seed)
rng = StableRNG(seed)

base_dir = "/home/ceoas/rathorek/projects/CStar/revision_exp/sampled_op/obs_xi_$(cons_arg)_$(SAMPLE_FREQ)"
plot_dir = "$base_dir/ts$(train_size)_seed$(seed)"
mkpath(plot_dir)

println("="^60)
println("CStar Comparison v3-sparse | Train: $train_size | Seed: $seed | Consumption: $cons_arg | SampleFreq: $SAMPLE_FREQ")
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
#   holling3  ->  x^2/(x^2+h^2)   [Tilman / type-III functional response]
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
# Simulation runs at full dense resolution.  Subsampling is applied after.
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
    data_n[1, :] .= max.(data_n[1, :], 0.0)   # x must be non-negative
    X_train_dense = data_n[:, 1:train_size]
    X_test        = data_n[1, train_size+1:train_size+HORIZON]
    return X_train_dense, X_test, data_n, data   # data = clean simulation
end

X_train_dense, X_test, data_n, data_clean = data_prep(train_size, rng)
c_real = collect(range(0.0, length=DATASIZE, stop=4.0))

# ── Sparse subsampling ────────────────────────────────────────────────────────
# obs_idx: which dense timesteps are observed (1-based, within train window)
# N_obs:   number of observed points
# X_train: [2 × N_obs] — sparse observed data (what UDE/GP/baselines see)
# c_real_obs, dip_regime_obs: labels at observed indices for evaluation
const obs_idx   = collect(1:SAMPLE_FREQ:train_size)   # e.g. [1,4,7,...] for freq=3
const N_obs     = length(obs_idx)
const X_train   = X_train_dense[:, obs_idx]           # 2 × N_obs

x_obs_train = X_train[1, :]    # noisy x at observed times  [N_obs]
i_obs_train = X_train[2, :]    # noisy i at observed times  [N_obs]

println("Data ready | dense_train=$train_size | obs_freq=$SAMPLE_FREQ | N_obs=$N_obs | test=$(length(X_test))")
println("obs_idx range: $(obs_idx[1]) … $(obs_idx[end])")

# ── Metrics ───────────────────────────────────────────────────────────────────
rmse(y, ŷ) = sqrt(mean((y .- ŷ).^2))
mae(y, ŷ)  = mean(abs.(y .- ŷ))
mape(y, ŷ) = mean(abs.((y .- ŷ) ./ (abs.(y) .+ 1e-10))) * 100

# ══════════════════════════════════════════════════════════════════════════════
# DIP-TEST REGIME DETECTOR  (unchanged from v2)
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
x_full_obs       = Float64.(data_n[1, :])
dip_regime_all, dip_sig, dip_var_thr = run_dip_detector(x_full_obs, c_real)
# For evaluation: labels at observed indices only
dip_regime_train     = dip_regime_all[1:train_size]        # dense (used in plots)
dip_regime_obs       = dip_regime_all[obs_idx]             # sparse (used in F1)
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

# ── Algebraic c reference ─────────────────────────────────────────────────────
# NOTE: algebraic_c uses only x (same as before — i is absorbed in residual).
# ── Algebraic c reference (sparse-aware) ─────────────────────────────────────
# When SAMPLE_FREQ > 1, consecutive observed x values are SAMPLE_FREQ steps
# apart in real time, so the finite-difference approximation of dx/dt changes.
# We treat the gap as SAMPLE_FREQ single-step Euler steps, which at first order
# gives: x[k+Δt] ≈ x[k] + Δt*(r·x·(1-x/K) - c·φ(x) + i·x)
# → c ≈ (r·x·(1-x/K) + x - x[k+Δt]/Δt·... )
# For simplicity we use the same algebraic inversion but divide numerator by Δt
# to account for the longer gap.  This is an approximation; the UDE/GP learn
# the exact multi-step residual.

function algebraic_c(x::AbstractVector{<:Real}, i_obs::AbstractVector{<:Real})
    xₜ  = Float64.(x[1:end-1]);  xₜ₁ = Float64.(x[2:end])
    iₜ  = Float64.(i_obs[1:end-1])
    num   = R_PARAM .* xₜ .* (1 .- xₜ./K_PARAM) .+ (1 .+ iₜ) .* xₜ .- xₜ₁
    denom = consumption.(xₜ)
    c_alg = num ./ denom
    c_alg[(xₜ .< 1e-8) .| (abs.(denom) .< 1e-10)] .= NaN
    return c_alg
end
alg_c = max.(algebraic_c(x_obs_train, i_obs_train), 0.0)


# ── Physics step (uses observed i directly) ───────────────────────────────────
# c is latent; i_val is the *observed* noisy i at that timestep.
physics_step_x_obs(x, c, i_val) =
    clamp(x + R_PARAM*x*(1 - x/K_PARAM) - c*consumption(x) + i_val*x, 1e-8, 1e4)

# ── GP top-level extensions ───────────────────────────────────────────────────
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::ColVecs) = map(m.f, eachcol(x.X))
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::RowVecs) = map(m.f, eachrow(x.X))

# GP over 2D input [x_norm, i_norm]
function construct_gp_2d(X_2d, params)
    kernel = params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ)
    return GP(x -> 0.0, kernel)(RowVecs(X_2d), params.var_noise)
end

# ── Shared helpers ────────────────────────────────────────────────────────────
base_res(method, t_secs) = Dict(
    "method"=>method, "seed"=>seed, "train_size"=>train_size,
    "sample_freq"=>SAMPLE_FREQ, "n_obs"=>N_obs,
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
    cols = ["method","seed","train_size","sample_freq","n_obs","r","K","h","time",
            "rmseC","rmseC_alg",
            "f1_1","f1_2","f1_3","macro_f1","weighted_f1",
            "f1_1_dip","f1_2_dip","f1_3_dip","macro_f1_dip","weighted_f1_dip",
            "rmse1","rmse5","rmse10","rmse12","rmse15",
            "mae1","mae5","mae10","mae12","mae15",
            "mape1","mape5","mape10","mape12","mape15",
            "x_forecast","c_predicted"]
    for c in cols
        c in names(df) || (df[!, c] = fill(missing, nrow(df)))
    end
    return select(df, cols)
end

# ── Plot constants ────────────────────────────────────────────────────────────
const METHOD_COLORS  = OrderedDict("UDE_obs"=>"#e41a1c","GP_obs"=>"#377eb8",
                                   "ARIMA"=>"#4daf4a","RW"=>"#984ea3","LSTM"=>"#ff7f00")
const METHOD_MARKERS = OrderedDict("UDE_obs"=>:circle,"GP_obs"=>:square,
                                   "ARIMA"=>:diamond,"RW"=>:utriangle,"LSTM"=>:star5)
const PLOT_MARGINS = (left_margin=8Plots.mm, bottom_margin=8Plots.mm,
                      right_margin=4Plots.mm, top_margin=4Plots.mm)

const REGIME_COLORS = [:steelblue, :darkorange, :firebrick]
const REGIME_NAMES  = ["stable_high", "flickering", "stable_low"]
const REGIME_ALPHA  = [0.25, 0.30, 0.25]

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

# ── Plot utilities (same as v2, updated method labels) ────────────────────────
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

function c_hat_plot(true_c, pred_c_dict, alg_c, dip_regime, train_size, tag, out_dir)
    T_c     = 1:train_size
    T_c_alg = 1:length(alg_c)
    y_lo = -0.5;  y_hi = 10.0

    p = plot(T_c, true_c,
             label="True c", color=:black, lw=2.5,
             xlabel="Time step", ylabel="Net harvesting pressure (c)",
             title="C Recovery + Dip Regimes | $tag",
             legend=:best, size=(1100,500), dpi=200; PLOT_MARGINS...)

    for (code, col, alpha) in zip(0:2, REGIME_COLORS, REGIME_ALPHA)
        segs = contiguous_segments(findall(i -> i<=train_size && dip_regime[i]==code, 1:train_size))
        isempty(segs) && continue
        for (k, (s, e)) in enumerate(segs)
            lbl = k == 1 ? "dip: $(REGIME_NAMES[code+1])" : nothing
            plot!(p, Shape([s, e, e, s], [y_lo, y_lo, y_hi, y_hi]);
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

function regime_overlay_plot(x_obs, c_real_full, dip_regime, train_size, tag, out_dir)
    gr()
    ts = 1:train_size
    det_regime = [c < CSTAR1 ? 0 : c < CSTAR2 ? 1 : 2 for c in c_real_full[1:train_size]]

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
    plot!(p1, ts, c_real_full[1:train_size]; color=:black, lw=2.0, label=nothing)

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
    plot!(p2, ts, x_obs[1:train_size]; color=:black, lw=1.4, label=nothing)

    p3 = plot(ylabel="regime source", xlabel="timestep",
              ylim=(-0.6, 1.6), legend=:topright,
              yticks=([0.0, 1.0], ["dip", "det"]),
              title="Regime agreement: det (top) vs dip (bottom)";
              PLOT_MARGINS...)
    for (code, col) in zip(0:2, REGIME_COLORS)
        for t in findall(==(code), det_regime)
            plot!(p3, [t-0.5, t+0.5], [1.0, 1.0]; color=col, lw=7, label="")
        end
        for t in findall(i -> i<=train_size && dip_regime[i]==code, 1:train_size)
            plot!(p3, [t-0.5, t+0.5], [0.0, 0.0]; color=col, lw=7, label="")
        end
    end
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
# METHOD 1: UDE_obs  (sparse-aware)
#
# uhat: 2 × N_obs  — latent state at observed timesteps only.
# Dynamical loss: for each consecutive observed pair (obs_idx[k], obs_idx[k+1]),
#   roll physics SAMPLE_FREQ single-steps starting from uhat[:, k].
#   The rolled x must match uhat[1, k+1]; c drifts by NN-predicted Δc per step.
# Observational loss: uhat[1, :] ≈ x_obs_train  (at observed indices only).
# ══════════════════════════════════════════════════════════════════════════════
function run_ude_obs(X_train, X_test, rng, dip_regime_obs)
    println("\n" * "="^60)
    println("METHOD: UDE_obs  (sparse freq=$SAMPLE_FREQ, N_obs=$N_obs)")
    println("="^60)

    ITERS = 1000
    lr    = 0.003f0

    NN = Lux.Chain(
        Lux.Dense(2,  8,  tanh),
        Lux.Dense(8,  32, tanh),
        Lux.Dense(32, 16, tanh),
        Lux.Dense(16, 8,  tanh),
        Lux.Dense(8,  1))

    NNparams, st = Lux.setup(rng, NN)

    x_mean = Float32(mean(X_train[1, :]))
    x_std  = Float32(std(X_train[1, :]))
    i_mean = Float32(mean(X_train[2, :]))
    i_std  = Float32(std(X_train[2, :]))

    normalize_xi(x, i_val) = Float32[
        (x     - x_mean) / (x_std + 1f-8),
        (i_val - i_mean) / (i_std + 1f-8)
    ]

    c_init = Float32.(rand(rng, 1, N_obs))
    uhat   = vcat(Float32.(X_train[1:1, :]), c_init)   # 2 × N_obs

    params0 = ComponentArray(NNparams=NNparams, uhat=uhat)

    l2_reg(NNp) = sum(sum(NNp[k].weight .^ 2) for k in keys(NNp))

    # i_obs_f32: observed noisy i at each observed timestep [N_obs]
    i_obs_f32 = Float32.(X_train[2, :])

    # One single-step physics advance: x,c → x_next, c_next
    # i_val is linearly interpolated between observed i values for interior steps.
    function single_step(x, c, i_val, NNp)
        inp    = normalize_xi(x, i_val)
        Δc     = NN(inp, NNp, st)[1][1]
        x_next = Float32(clamp(
            x + R_PARAM*x*(1 - x/K_PARAM) - c*consumption(x) + i_val*x,
            1f-8, 1f4))
        c_next = c + Δc
        return x_next, c_next
    end

    # Roll SAMPLE_FREQ steps from observed index k to k+1.
    # i is linearly interpolated between i_obs[k] and i_obs[k+1].
    function rollout(uhat_k, i_k, i_k1, NNp)
        x, c = uhat_k[1], uhat_k[2]
        for s in 0:(SAMPLE_FREQ-1)
            α     = Float32(s) / Float32(SAMPLE_FREQ)
            i_val = (1f0 - α)*i_k + α*i_k1
            x, c  = single_step(x, c, i_val, NNp)
        end
        return Float32[x, c]
    end

    function loss(params, data)
        𝐮 = params.uhat
        # Dynamical loss: rolled state at k+1 ≈ uhat[:, k+1]
        L_dyn = sum(
            sum((𝐮[:, k+1] .- rollout(𝐮[:, k],
                                       i_obs_f32[k], i_obs_f32[k+1],
                                       params.NNparams)).^2)
            for k in 1:(N_obs-1))
        # Observational fidelity at observed timesteps
        L_obs = sum((Float32.(data[1:1, :]) .- 𝐮[1:1, :]).^2)
        total = 0.4f0*L_dyn + 0.4f0*L_obs + 0.2f0*l2_reg(params.NNparams)
        return isfinite(total) ? total : 1f10
    end

    t_start = now()
    losses  = Float64[]
    callback(state, l) = begin
        push!(losses, l); n = length(losses)
        n % 500 == 0 && println("  UDE_obs iter=$n | loss=$(round(l,digits=5))")
        stop = !isfinite(l) || l > 1f8
        stop && println("  UDE_obs divergence stop at iter $n")
        return stop
    end

    adtype = Optimization.AutoZygote()
    optf   = Optimization.OptimizationFunction((p, _) -> loss(p, X_train), adtype)
    prob1  = Optimization.OptimizationProblem(optf, params0)
    sol1   = Optimization.solve(prob1, Optimisers.ADAM(lr), callback=callback, maxiters=ITERS)

    lbfgs_losses = Float64[]
    cb2(state, l) = begin
        push!(lbfgs_losses, l); n = length(lbfgs_losses)
        n % 100 == 0 && println("  L-BFGS iter=$n | loss=$(round(l,digits=5))")
        stop = !isfinite(l) || l > 1f8 || (n > 1 && abs(lbfgs_losses[end-1] - l) < F_TOL)
        stop && println("  L-BFGS stop at iter $n")
        return stop
    end
    prob2 = Optimization.OptimizationProblem(optf, sol1.u)
    sol2  = Optimization.solve(prob2, LBFGS(linesearch=BackTracking()),
                               callback=cb2, g_abstol=5f-4)
    append!(losses, lbfgs_losses)
    t_end = now()
    println("UDE_obs done in $(canonicalize(t_end-t_start)) | $(length(losses)) iters | loss=$(round(losses[end],digits=5))")

    # ── Forecast (dense, step-by-step from last observed state) ───────────────
    i_last = Float32(i_obs_f32[end])
    preds  = fill(NaN32, 2, HORIZON + 1)
    preds[:, 1] = [Float32(X_train[1, end]), sol2.u.uhat[2, end]]

    i_fc = i_last
    for t in 2:(HORIZON + 1)
        x_prev, c_prev = preds[1, t-1], preds[2, t-1]
        x_next, c_next = single_step(x_prev, c_prev, i_fc, sol2.u.NNparams)
        preds[:, t] = [x_next, c_next]
        i_fc = Float32(1 - 1/T_PARAM) * i_fc
    end

    pred_x = Float64.(preds[1, 2:end])
    for k in eachindex(pred_x)
        isfinite(pred_x[k]) || (pred_x[k] = k > 1 ? pred_x[k-1] : X_train[1, end])
    end

    pred_c   = Float64.(sol2.u.uhat[2, :])          # at observed indices
    true_c   = c_real[obs_idx]                       # c_real at observed indices
    alg_idx  = findall(isfinite, alg_c)
    c_rmse   = round(rmse(true_c, pred_c),                  digits=5)
    alg_rmse = round(rmse(true_c[alg_idx], alg_c[alg_idx]), digits=5)
    println("UDE_obs C-RMSE: $c_rmse | Alg C-RMSE: $alg_rmse")

    det_metrics = classification_metrics_det(true_c, pred_c, "UDE_obs")
    dip_metrics = classification_metrics_dip(dip_regime_obs, pred_c, "UDE_obs")

    res = merge!(base_res("UDE_obs", Dates.value(t_end-t_start)/1000),
                 det_metrics, dip_metrics,
                 Dict("rmseC"=>c_rmse, "rmseC_alg"=>alg_rmse,
                      "x_forecast" =>string(round.(pred_x, digits=5)),
                      "c_predicted"=>string(round.(pred_c, digits=5))))
    fill_metrics!(res, pred_x, X_test)
    return DataFrame(res), pred_x, pred_c, losses
end

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2: GP_obs  (sparse-aware)
#
# uhat: 2 × N_obs  — [x̂; ĉ] at observed indices only.
# GP input: [x_norm, i_norm] at obs k → target Δc = ĉ[k+1] - ĉ[k]
# Dynamical loss: multi-step rollout bridging obs k → k+1 (SAMPLE_FREQ steps).
#   i is linearly interpolated between i_obs[k] and i_obs[k+1].
# ══════════════════════════════════════════════════════════════════════════════
function run_gp_obs(X_train, X_test, rng, dip_regime_obs)
    println("\n" * "="^60)
    println("METHOD: GP_obs  (sparse freq=$SAMPLE_FREQ, N_obs=$N_obs)")
    println("="^60)

    MAX_ITERS = 3000
    LR        = 0.003

    x_mean = mean(X_train[1, :])
    x_std  = std(X_train[1, :])
    i_mean = mean(X_train[2, :])
    i_std  = std(X_train[2, :])

    normalize_x(x)  = (x    - x_mean) / (x_std + 1e-8)
    normalize_i(iv) = (iv   - i_mean) / (i_std + 1e-8)

    σ²_k = max(var(x_obs_train), 1e-4)

    flat_initial_params, unflatten = flatten((
        var_kernel = positive(σ²_k),
        λ          = positive(0.5),
        var_noise  = positive(0.1σ²_k),
        uhat       = vcat(x_obs_train', rand(rng, 1, N_obs)),
    ))
    unpack = ParameterHandling.value ∘ unflatten

    i_obs_f64 = Float64.(i_obs_train)   # fixed, kept out of AD graph

    # Multi-step rollout (pure physics, no GP) bridging obs k → k+1
    function rollout_x(x_k, c_k, i_k, i_k1)
        x = x_k
        for s in 0:(SAMPLE_FREQ-1)
            α   = s / SAMPLE_FREQ
            i_s = (1 - α)*i_k + α*i_k1
            x   = physics_step_x_obs(x, c_k, i_s)   # c held fixed during gap
        end
        return x
    end

    function objective(params)
        𝐱  = params.uhat[1, :]
        𝐜  = params.uhat[2, :]
        X_2d = hcat(normalize_x.(𝐱[1:end-1]), normalize_i.(i_obs_f64[1:end-1]))
        lml  = -logpdf(construct_gp_2d(X_2d, params), diff(𝐜))
        L_dyn = sum(
            (𝐱[k+1] - rollout_x(𝐱[k], 𝐜[k], i_obs_f64[k], i_obs_f64[k+1]))^2
            for k in 1:(N_obs-1))
        L_obs = sum((x_obs_train .- 𝐱).^2)
        return 0.5*L_dyn + 0.5*L_obs + lml
    end

    opt         = Optimise.ADAM(LR)
    flat_params = deepcopy(flat_initial_params)
    losses      = Float64[]
    t_start     = now()

    for epoch in 1:MAX_ITERS
        if epoch % 500 == 0
            opt.eta /= 2.0
            # println("  Reduced LR to $(opt.eta)")
        end
        grads      = Zygote.gradient(θ -> objective(unpack(θ)), flat_params)
        Optimise.update!(opt, flat_params, grads[1])
        loss_value = objective(unpack(flat_params))
        push!(losses, loss_value)
        epoch % 500 == 0 &&
            println("  GP_obs iter=$epoch | loss=$(round(loss_value,digits=5)) | gnorm=$(round(norm(grads),digits=5)) | LR to $(opt.eta)")
        length(losses) > 1 && abs(losses[end-1] - loss_value) < F_TOL &&
            (println("  GP_obs early stop at iter $epoch"); break)
    end
    t_end = now()
    println("GP_obs done in $(canonicalize(t_end-t_start))")

    final_params = unpack(flat_params)
    c_lat_final  = final_params.uhat[2, :]
    x_lat_final  = final_params.uhat[1, :]

    X_2d_final  = hcat(normalize_x.(x_lat_final[1:end-1]),
                       normalize_i.(i_obs_f64[1:end-1]))
    posterior_c = posterior(construct_gp_2d(X_2d_final, final_params), diff(c_lat_final))

    function predict_Δc(x, i_val)
        xi = reshape([normalize_x(x), normalize_i(i_val)], 1, 2)
        return mean(marginals(posterior_c(RowVecs(xi)))[1])
    end

    # ── Forecast (dense, step-by-step from last obs) ───────────────────────────
    i_fc  = i_obs_f64[end]
    preds = fill(NaN, 2, HORIZON + 1)
    preds[:, 1] = [X_train[1, end], c_lat_final[end]]

    for t in 2:(HORIZON + 1)
        x_prev, c_prev = preds[1, t-1], preds[2, t-1]
        x_next  = physics_step_x_obs(x_prev, c_prev, i_fc)
        delta_c = predict_Δc(x_prev, i_fc)
        c_next  = c_prev + delta_c
        preds[:, t] = [x_next, c_next]
        i_fc    = (1 - 1/T_PARAM) * i_fc
    end

    pred_x = preds[1, 2:end]
    for k in eachindex(pred_x)
        isfinite(pred_x[k]) || (pred_x[k] = k > 1 ? pred_x[k-1] : X_train[1, end])
    end

    true_c   = c_real[obs_idx]
    alg_idx  = findall(isfinite, alg_c)
    c_rmse   = round(rmse(true_c, c_lat_final),               digits=5)
    alg_rmse = round(rmse(true_c[alg_idx], alg_c[alg_idx]),   digits=5)
    println("GP_obs C-RMSE: $c_rmse | Alg C-RMSE: $alg_rmse")

    det_metrics = classification_metrics_det(true_c, c_lat_final, "GP_obs")
    dip_metrics = classification_metrics_dip(dip_regime_obs, c_lat_final, "GP_obs")

    res = merge!(base_res("GP_obs", Dates.value(t_end-t_start)/1000),
                 det_metrics, dip_metrics,
                 Dict("rmseC"=>c_rmse, "rmseC_alg"=>alg_rmse,
                      "x_forecast" =>string(round.(pred_x,      digits=5)),
                      "c_predicted"=>string(round.(c_lat_final, digits=5))))
    fill_metrics!(res, pred_x, X_test)
    return DataFrame(res), pred_x, c_lat_final, losses
end

# ══════════════════════════════════════════════════════════════════════════════
# METHODS 3 & 4: ARIMA and RandomWalk  (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════
function run_baseline(name, x_train, X_test, build_model)
    println("\n" * "="^60 * "\nMETHOD: $name\n" * "="^60)
    t0  = now()
    res = merge!(base_res(name, 0.0),
                 Dict("rmseC"=>missing, "rmseC_alg"=>missing,
                      "c_predicted"=>missing,
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
# METHOD 5: LSTM  (unchanged from v2 — x-only, no knowledge of i or c)
# ══════════════════════════════════════════════════════════════════════════════
function run_lstm(X_train, X_test, rng)
    println("\n" * "="^60)
    println("METHOD: LSTM")
    println("="^60)

    SEQ_LEN = 10
    ITERS   = 15000
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
                 Dict("rmseC"=>missing,"rmseC_alg"=>missing,
                      "c_predicted"=>missing,
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
ude_df,  ude_pred_x,  ude_pred_c,  ude_losses  = run_ude_obs(X_train, X_test, rng, dip_regime_obs)
gp_df,   gp_pred_x,   gp_pred_c,   gp_losses   = run_gp_obs(X_train, X_test, rng, dip_regime_obs)
arima_df, arima_pred_x                          = run_arima(x_obs_train, X_test)
rw_df,   rw_pred_x                              = run_randomwalk(x_obs_train, X_test)
lstm_df, lstm_pred_x, lstm_losses               = run_lstm(X_train, X_test, rng)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
all_dfs = [ude_df, gp_df, arima_df, rw_df, lstm_df]
std_dfs = [standardize_df(df) for df in all_dfs]

tag      = "train$(train_size)_seed$(seed)_freq$(SAMPLE_FREQ)"
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
for (lbl, df) in zip(["UDE_obs","GP_obs","ARIMA","RW","LSTM"], std_dfs)
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
gr()

# Forecast plot: use dense x as background context, sparse obs as scatter overlay
let
    x_dense = Float64.(data_n[1, 1:train_size])   # full dense noisy x (background)
    z       = max(1, train_size - 50)
    T_te    = train_size:train_size+HORIZON

    p = plot(z:train_size, x_dense[z:end],
             label="Dense x (sim)", color=:lightgray, lw=1.5, alpha=0.6,
             xlabel="Time step", ylabel="Abundance (x)",
             title="X Forecast | $tag",
             legend=:best, size=(1100,500), dpi=200; PLOT_MARGINS...)
    # Mark the actual sparse observations
    obs_in_zoom = obs_idx[obs_idx .>= z]
    scatter!(p, obs_in_zoom, x_dense[obs_in_zoom],
             label="Observed (freq=$SAMPLE_FREQ)", color=:black, ms=4, msw=0)
    plot!(p, T_te, vcat(x_dense[end], X_test),
          label="True x (test)", color=:black, lw=2.5)
    for (lbl, fcast) in OrderedDict("UDE_obs"=>ude_pred_x,"GP_obs"=>gp_pred_x,
                                     "ARIMA"=>arima_pred_x,"RW"=>rw_pred_x,"LSTM"=>lstm_pred_x)
        plot!(p, T_te, vcat(x_dense[end], fcast),
              label=lbl, color=get(METHOD_COLORS,lbl,"#333333"), lw=2, ls=:dash,
              marker=get(METHOD_MARKERS,lbl,:circle), ms=4)
    end
    vline!(p, [train_size], color=:black, ls=:dot, lw=1.5, label=false)
    savefig(p, "$plot_dir/X_forecast_$(tag).png")
end

# C-hat plot: true c (dense line) vs predicted c (at observed indices only)
let
    p = plot(1:train_size, c_real[1:train_size],
             label="True c", color=:black, lw=2.5,
             xlabel="Time step", ylabel="Net harvesting pressure (c)",
             title="C Recovery | $tag",
             legend=:best, size=(1100,500), dpi=200; PLOT_MARGINS...)
    y_lo, y_hi = -0.5, 10.0

    # Dip regime shading (dense, for full visual context)
    for (code, col, alpha) in zip(0:2, REGIME_COLORS, REGIME_ALPHA)
        segs = contiguous_segments(
            findall(i -> i<=train_size && dip_regime_train[i]==code, 1:train_size))
        for (k, (s, e)) in enumerate(segs)
            lbl = k == 1 ? "dip: $(REGIME_NAMES[code+1])" : nothing
            plot!(p, Shape([s,e,e,s],[y_lo,y_lo,y_hi,y_hi]);
                  color=col, alpha=alpha, lw=0, label=lbl)
        end
    end

    hline!(p, [0.0],    color=:steelblue, ls=:dash, lw=1.0, label="c = 0")
    hline!(p, [CSTAR1], color="#e6ab02",  ls=:dot,  lw=1.5,
           label="C*₁=$(round(CSTAR1,digits=3))")
    hline!(p, [CSTAR2], color="#d95f02",  ls=:dot,  lw=1.5,
           label="C*₂=$(round(CSTAR2,digits=3))")

    # Algebraic c and predicted c — plotted at obs_idx positions
    alg_full = fill(NaN, train_size)
    alg_full[obs_idx[1:length(alg_c)]] .= alg_c   # alg_c has N_obs-1 entries
    plot!(p, 1:train_size, alg_full,
          label="Algebraic c", color=:gray, lw=1, ls=:dot, alpha=0.7)

    for (lbl, pc) in OrderedDict("UDE_obs"=>ude_pred_c, "GP_obs"=>gp_pred_c)
        isnothing(pc) && continue
        pred_full = fill(NaN, train_size)
        pred_full[obs_idx] .= pc
        plot!(p, 1:train_size, pred_full,
              label=lbl, color=get(METHOD_COLORS,lbl,"#333333"),
              lw=0, marker=get(METHOD_MARKERS,lbl,:circle), ms=4, alpha=0.85)
    end
    ylims!(p, y_lo, y_hi)
    savefig(p, "$plot_dir/C_hat_$(tag).png")
end

# Regime overlay: always use dense x for visual fidelity
regime_overlay_plot(Float64.(data_n[1, 1:train_size]), c_real, dip_regime_train,
                    train_size, tag, plot_dir)

rmse_window_plot(
    OrderedDict("UDE_obs"=>std_dfs[1],"GP_obs"=>std_dfs[2],
                "ARIMA"=>std_dfs[3],"RW"=>std_dfs[4],"LSTM"=>std_dfs[5]),
    tag, plot_dir)

combined_loss_plot(
    OrderedDict("UDE_obs"=>ude_losses,"GP_obs"=>gp_losses,"LSTM"=>lstm_losses),
    tag, plot_dir)

println("\nAll plots saved for $tag")
println("Script completed successfully!")