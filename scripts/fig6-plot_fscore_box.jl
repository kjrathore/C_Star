using Pkg
Pkg.activate("env_c")

using CSV, DataFrames, StatsPlots, CategoricalArrays, Plots, Colors
using Plots.PlotMeasures

# --- Setup ---
input_file  = "revision_csv/regime_results_123.csv"
output_plot = "revision_plots/box_f1_regimes.png"

fnt_guide  = 18
fnt_tick   = 14
fnt_legend = 13

# --- Load & filter ---
all_data = CSV.read(input_file, DataFrame)
if "training_size" in names(all_data)
    rename!(all_data, :training_size => :train_size)
end

df = all_data[all_data.train_size .== 300, :]

method_map   = Dict("Alg_c" => "Algebraic", "UDE-NN" => "UDE-NN", "GP" => "GP")
method_order = ["Algebraic", "GP", "UDE-NN"]

df = df[in.(df.method, Ref(keys(method_map))), :]
df[!, :method] = [method_map[m] for m in df.method]

# --- Colors with alpha ---
# Row vector (1×N) is required by groupedboxplot for per-group color assignment
raw_colors  = [:orange, :purple, :green]
alpha_val   = 0.75
box_colors  = [RGBA(red(c), green(c), blue(c), alpha_val)
               for c in parse.(Colorant, string.(raw_colors))]
color_row   = permutedims(box_colors)   # 1×3 row vector

# --- Reshape ---
df_long = stack(df, [:f1_1, :f1_2, :f1_3], :method)
rename!(df_long, :variable => :regime, :value => :f1)
df_long = dropmissing(df_long, :f1)

regime_label = Dict("f1_1" => "Regime I", "f1_2" => "Regime II", "f1_3" => "Regime III")
df_long[!, :regime] = [regime_label[string(r)] for r in df_long.regime]

regime_order = ["Regime I", "Regime II", "Regime III"]
df_long.regime = categorical(df_long.regime)
levels!(df_long.regime, regime_order)

df_long.method = categorical(df_long.method)
levels!(df_long.method, method_order)

# Sort so groupedboxplot sees consistent ordering
sort!(df_long, [:regime, :method])

# --- Plot ---
p = @df df_long groupedboxplot(
    :regime, :f1,
    group             = :method,
    bar_width         = 0.75,
    linewidth         = 1.5,
    color             = color_row,
    fillalpha         = 0.65,
    label             = permutedims(method_order),
    whiskerwidth      = 0.5,
    mediancolor       = :black,
    outliers          = true,
    markerstrokewidth = 1.0,
    xlabel            = "Regime",
    ylabel            = "F₁ score",
    framestyle        = :box,
    legend            = :bottomleft,
    legendfont        = font(fnt_legend),
    guidefont         = font(fnt_guide),
    tickfont          = font(fnt_tick),
    foreground_color_legend = nothing,
    background_color_legend = :white,
    ylims             = (0, 1.05),
    dpi               = 400,
    size              = (900, 550),
    left_margin       = 10mm,
    right_margin      = 5mm,
    top_margin        = 5mm,
    bottom_margin     = 8mm,
)

savefig(p, output_plot)
println("Saved → $output_plot")