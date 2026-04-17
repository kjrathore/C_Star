# ============================================================
# analyze_comp_op.jl
# Usage: julia analyze_comp_op.jl [path/to/comp_op]
#
# 1. Reports missing/extra seeds per results CSV
# 2. Prints per-consumption-type summary tables (mean ± std)
# 3. Saves comp_op/comp_op_summary.csv
# ============================================================

using Pkg
Pkg.activate("/home/ceoas/rathorek/projects/CStar/env_c")

using CSV, DataFrames, Statistics, Printf, Dates

# ── config ────────────────────────────────────────────────────────────────────
const SEEDS = Set([
    26058,46189,77471,38315,97565,60590,11007,48073,69686,1322,
    24415,15318,16782,96004,8694,28757,60467,29623,86366,18423,
    46984,51568,67883,89255,61917,30563,54133,89053,37085,6871,
    4150,22918,52212,28020,69377,49858,96331,16025,93091,79876,
    88243,66214,7296,91809,91328,19561,9503,44769,27272,93050
])

const METHODS = ["UDE_i", "GP_i", "ARIMA", "RW", "LSTM"]

const METRIC_GROUPS = [
    "Latent c recovery" => ["rmseC", "rmseC_alg", "rmseI"],
    "Regimewise F1 (c*)"=> ["f1_1", "f1_2", "f1_3"],
    "Regime F1 (c*)"    => ["macro_f1", "weighted_f1"],
    "Regime F1 (dip)"   => ["macro_f1_dip", "weighted_f1_dip"],
    "Forecast RMSE"     => ["rmse1", "rmse5", "rmse10", "rmse15"],
    "Forecast MAE"      => ["mae1",  "mae5",  "mae10",  "mae15"],
    "Runtime (s)"       => ["time"],
]

# ── helpers ───────────────────────────────────────────────────────────────────
μσ(v) = isempty(v) ? (NaN, NaN) : (mean(v), std(v))

fmt_cell(μ, σ) = isnan(μ) ? lpad("—", 16) : @sprintf("  %7.4f±%6.4f", μ, σ)

function print_sep(char, n=70) println(char ^ n) end

function load_csv(path)
    try
        # Drop trailing array columns (x_forecast, c_predicted, i_predicted) which
        # contain embedded commas and confuse the column count heuristic.
        df = CSV.read(path, DataFrame;
                      strict=false,
                      missingstring=["", "NA", "NaN"],
                      drop=["x_forecast", "c_predicted", "i_predicted"],
                      types=Dict("seed" => Int64))
        return df
    catch e
        # If drop kwarg fails (older CSV.jl), fall back and select only scalar cols
        try
            df = CSV.read(path, DataFrame; strict=false, missingstring=["", "NA", "NaN"])
            array_cols = ["x_forecast", "c_predicted", "i_predicted"]
            return select(df, [n for n in names(df) if n ∉ array_cols])
        catch e2
            @warn "Failed to read $path: $e2"
            return nothing
        end
    end
end

# ── main ──────────────────────────────────────────────────────────────────────
base = length(ARGS) >= 1 ? ARGS[1] : "comp_op"

subdirs = sort(filter(isdir, readdir(base; join=true)))

# ── 1. Seed coverage ──────────────────────────────────────────────────────────
println("=" ^ 80)
println("  SEED COVERAGE REPORT")
println("=" ^ 80)

dfs = Dict{String,DataFrame}()
 
for sd in subdirs
    csvs = filter(f -> endswith(f, ".csv") && startswith(basename(f), "results"),
                  readdir(sd; join=true))
    if isempty(csvs)
        println("\n$(basename(sd)): NO CSV FOUND"); continue
    end
    df = load_csv(csvs[1])
    isnothing(df) && continue
 
    present = "seed" ∈ names(df) ?
              Set(skipmissing(df.seed)) : Set{Int}()
    missing_seeds = sort(collect(setdiff(SEEDS, present)))
    extra_seeds   = sort(collect(setdiff(present, SEEDS)))
 
    println("\n" * "─" ^ 70)
    println("  $(basename(sd))  [$(basename(csvs[1]))]")
    println("  Seeds found  : $(length(present)) / $(length(SEEDS))")
    if isempty(missing_seeds)
        println("  ✓ All $(length(SEEDS)) seeds present")
    else
        chunks = [missing_seeds[i:min(i+9, end)] for i in 1:10:length(missing_seeds)]
        println("  MISSING ($(length(missing_seeds))) : $(chunks[1])")
        for c in chunks[2:end]
            println("               $c")
        end
    end
    isempty(extra_seeds) || println("  Extra        : $extra_seeds")
 
    dfs[basename(sd)] = df
end

# ── 2. Summary tables ─────────────────────────────────────────────────────────
println("\n\n" * "=" ^ 80)
println("  SUMMARY TABLES  (mean ± std across seeds, per method)")
println("=" ^ 80)

all_rows = DataFrame()

for ctype in sort(collect(keys(dfs)))
    df = dfs[ctype]
    col_names = names(df)

    println("\n" * "═" ^ 70)
    println("  $ctype")
    println("═" ^ 70)

    # discover methods from data; prefer METHODS order, append any extras
    found_methods = unique(skipmissing(df[!, :method]))
    ordered = vcat(filter(m -> m ∈ found_methods, METHODS),
                   filter(m -> m ∉ METHODS, found_methods))
 
    meq(row, m) = !ismissing(row.method) && row.method == m
 
    # one row per method — accumulate all metrics across groups, push once
    method_rows = Dict(m => Dict{String,Any}(
                            "consumption_type" => ctype,
                            "method"           => m,
                            "n_seeds"          => nrow(filter(r -> meq(r, m), df)))
                       for m in ordered)

    for (gname, metrics) in METRIC_GROUPS
        avail = filter(m -> m ∈ col_names, metrics)
        isempty(avail) && continue
 
        println("\n  $gname")
        header = rpad("  Method", 12) * join(lpad(m, 16) for m in avail)
        println(header)
        println("  " * "─" ^ (length(header) - 2))
 
        for method in ordered
            sub = filter(r -> meq(r, method), df)
            print(rpad("  $method", 12))
            for m in avail
                parsed = tryparse.(Float64, string.(sub[!, m]))
                vals   = filter(!isnan, Float64.(filter(!isnothing, parsed)))
                μ, σ = μσ(vals)
                print(fmt_cell(μ, σ))
                method_rows[method][m] = isnan(μ) ? "" : @sprintf("%.4f±%.4f", μ, σ)
            end
            println()
        end
    end

    ctype_rows = DataFrame()
    for method in ordered
        push!(ctype_rows, method_rows[method]; cols=:union)
    end
    append!(all_rows, ctype_rows; cols=:union)
end

# ── 3. Save summary CSV ───────────────────────────────────────────────────────
out_path = joinpath(base, "comp_op_summary.csv")
if nrow(all_rows) > 0
    CSV.write(out_path, all_rows)
    println("\n\nSaved summary → $out_path")
else
    println("\nNo data to save.")
end