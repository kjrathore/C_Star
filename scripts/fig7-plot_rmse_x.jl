using Pkg
Pkg.activate("env_c")

using CSV, DataFrames, StatsPlots, CategoricalArrays, Plots
using HypothesisTests: MannWhitneyUTest, pvalue
using Statistics
using Plots.PlotMeasures
using Colors

# 1. Setup
input_file  = "revision_csv/regime_results_123.csv"
output_plot = "revision_plots/x_rmsebox.png"

# --- FONT SETTINGS ---
fnt_guide  = 16
fnt_tick   = 16
fnt_title  = 24
fnt_legend = 16

# --- ORIGINAL COLORS WITH ALPHA ---
raw_colors    = [:blue, :orange, :olive, :purple, :green, :cyan]
alpha_val     = 0.75
custom_colors = [RGBA(red(c), green(c), blue(c), alpha_val)
                 for c in parse.(Colorant, string.(raw_colors))]
all_data = CSV.read(input_file, DataFrame)
if "training_size" in names(all_data)
    rename!(all_data, :training_size => :train_size)
end

all_data = dropmissing(all_data, :method)
for col in [:rmse1, :rmse5, :rmse10, :rmse15]
    if col in names(all_data)
        all_data[!, col] = [tryparse(Float64, string(x)) for x in all_data[!, col]]
    end
end

train_sizes  = [70, 220, 300]
horizons     = [1, 5, 10, 15]
method_order = ["ARIMA", "RandomWalk", "LSTM", "GP", "UDE-NN"]

plots = []

for (i, t_size) in enumerate(train_sizes)
    df_size = all_data[all_data.train_size .== t_size, :]
    isempty(df_size) && continue

    present_methods = intersect(method_order, unique(df_size.method))
    counts = [nrow(df_size[df_size.method .== m, :]) for m in present_methods]
    isempty(counts) || minimum(counts) < 2 && continue

    n_min  = minimum(counts)
    df_sub = DataFrame()
    for meth in present_methods
        append!(df_sub, df_size[df_size.method .== meth, :][1:n_min, :])
    end

    plot_df = DataFrame(horizon=String[], value=Float64[], method=String[])
    for h in horizons
        col = Symbol("rmse$h")
        for meth in present_methods
            vals = collect(skipmissing(df_sub[df_sub.method .== meth, col]))
            vals = filter(x -> x > 0, vals)
            isempty(vals) && continue
            append!(plot_df, DataFrame(
                horizon = fill(string(h), length(vals)),
                value   = vals,
                method  = fill(string(meth), length(vals))
            ))
        end
    end

    isempty(plot_df) && continue

    plot_df.method  = categorical(plot_df.method,  levels=method_order, ordered=true)
    plot_df.horizon = categorical(plot_df.horizon, levels=string.(horizons), ordered=true)

    sub_levels  = levels(plot_df.method)
    sub_palette = [custom_colors[findfirst(==(val), method_order)] for val in sub_levels]

    p = groupedboxplot(
        plot_df.horizon,
        plot_df.value,
        group         = plot_df.method,
        title         = "N = $t_size",
        xlabel        = "Forecast horizon",
        ylabel        = (i == 1) ? "RMSE" : "",
        legend        = false,
        yscale        = :log10,
        bar_width     = 0.85,        # wider bars → more spread within group
        palette       = sub_palette,
        linewidth     = 0.8,
        mediancolor   = :white,
        whiskerwidth  = 0.5,
        markerstrokewidth = 0,
        guidefont     = font(fnt_guide),
        tickfont      = font(fnt_tick),
        titlefont     = font(fnt_title),
        left_margin   = (i == 1) ? 8mm : 0mm,
        right_margin  = 0mm,
        top_margin    = 2mm,
        bottom_margin = 6mm,
    )
    push!(plots, p)
end

if !isempty(plots)
    legend_p = plot(framestyle=:none, ticks=nothing, grid=false,
                    background_color=:transparent)
    for (idx, meth) in enumerate(method_order)
        scatter!(legend_p, [NaN], [NaN],
            label             = meth,
            color             = custom_colors[idx],
            marker            = :rect,
            markersize        = 10,
            markerstrokewidth = 0,
            legendcolumns     = length(method_order),
            legend            = :top,
            legendfont        = font(fnt_legend),
            foreground_color_legend  = nothing,
            background_color_legend  = :transparent,
        )
    end

    # FIX: h_spacing reduces horizontal gap between subplots
    combined = plot(
        plots...,
        layout        = (1, length(plots)),
        link          = :y,
        size          = (1600, 520),
        left_margin   = 0mm,
        right_margin  = 0mm,
    )

    final = plot(
        combined,
        legend_p,
        layout        = @layout([a{0.87h}; b{0.13h}]),
        size          = (1400, 650),
        left_margin   = 10mm,
        right_margin  = 6mm,
        top_margin    = 4mm,
        bottom_margin = 2mm,
    )

    savefig(final, output_plot)
    println("Saved → $output_plot")
end