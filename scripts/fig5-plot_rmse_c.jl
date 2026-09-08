using Pkg
Pkg.activate("env_c")

using CSV, DataFrames, StatsPlots, CategoricalArrays, Plots, Colors
using Plots.PlotMeasures

# --- Setup ---
input_file  = "revision_csv/regime_results_123.csv"
output_plot = "revision_plots/violin_rmseC.png"

fnt_guide  = 18
fnt_tick   = 14
fnt_title  = 20
fnt_legend = 22

# --- Load ---
all_data = CSV.read(input_file, DataFrame)
if "training_size" in names(all_data)
    rename!(all_data, :training_size => :train_size)
end

method_map   = Dict("Alg_c" => "Algebraic", "UDE-NN" => "UDE-NN", "GP" => "GP")
method_order = ["Algebraic", "GP", "UDE-NN"]

all_data = all_data[in.(all_data.method, Ref(keys(method_map))), :]
all_data[!, :method] = [method_map[m] for m in all_data.method]

# Clean rmseC
all_data[!, :rmseC] = [tryparse(Float64, string(x)) for x in all_data[!, :rmseC]]
all_data = dropmissing(all_data, :rmseC)
all_data = all_data[all_data.rmseC .> 0, :]

# --- Colors with alpha ---
raw_colors  = [:orange, :purple, :green]
alpha_fill   = 0.65
alpha_box    = 0.95
fill_colors  = [RGBA(red(c), green(c), blue(c), alpha_fill)
                for c in parse.(Colorant, string.(raw_colors))]
box_colors   = [RGBA(red(c), green(c), blue(c), alpha_box)
                for c in parse.(Colorant, string.(raw_colors))]

train_sizes = [70, 220, 300]
plots = []

for (i, t_size) in enumerate(train_sizes)
    df = all_data[all_data.train_size .== t_size, :]
    isempty(df) && continue

    df.method = categorical(df.method)
    levels!(df.method, method_order)

    p = plot(
        title         = "N = $t_size",
        xlabel        = "Method",
        ylabel        = (i == 1) ? "RMSE" : "",
        framestyle    = :box,
        legend        = false,
        guidefont     = font(fnt_guide),
        tickfont      = font(fnt_tick),
        titlefont     = font(fnt_title),
        left_margin   = (i == 1) ? 10mm : 2mm,
        right_margin  = 2mm,
        top_margin    = 4mm,
        bottom_margin = 8mm,
    )

    for (idx, meth) in enumerate(method_order)
        df_m = df[df.method .== meth, :]
        isempty(df_m) && continue

        # Violin
        @df df_m violin!(p,
            :method, :rmseC,
            side          = :both,
            linewidth     = 1.5,
            color         = fill_colors[idx],
            label         = false,
            # markerstrokewidth = 1.5,
            yscale        = :log10,
        )

        # Boxplot overlay
        @df df_m boxplot!(p,
            :method, :rmseC,
            bar_width         = 0.08,
            linewidth         = 1.2,
            color             = box_colors[idx],
            fillalpha         = 0.0,
            label             = false,
            whiskerwidth      = 0.4,
            mediancolor       = :black,
            outliers          = true,
            outliercolor      = box_colors[idx],
            markerstrokewidth = 1.5,
            markersize        = 3,
            yscale        = :log10,
        )
    end

    push!(plots, p)
end

# --- Legend panel ---
legend_p = plot(framestyle=:none, ticks=nothing, grid=false,
                background_color=:transparent)
for (idx, meth) in enumerate(method_order)
    scatter!(legend_p, [NaN], [NaN],
        label             = meth,
        color             = fill_colors[idx],
        marker            = :rect,
        markersize        = 10,
        markerstrokewidth = 0,
        legendcolumns     = length(method_order),
        legend            = :top,
        legendfont        = font(fnt_legend),
        foreground_color_legend = nothing,
        background_color_legend = :transparent,
    )
end

combined = plot(
    plots...,
    layout        = (1, length(plots)),
    # link          = :y,
    size          = (1400, 500),
)

final = plot(
    combined,
    legend_p,
    layout        = @layout([a{0.87h}; b{0.13h}]),
    size          = (1400, 620),
    left_margin   = 10mm,
    right_margin  = 5mm,
    top_margin    = 4mm,
    bottom_margin = 2mm,
    dpi           = 400,
)

savefig(final, output_plot)
println("Saved → $output_plot")