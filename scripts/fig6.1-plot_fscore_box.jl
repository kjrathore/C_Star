using Pkg
Pkg.activate("C:/Users/kj_ra/Box/Ecology/Project_CStar/HPC_codes/discrete_dym/env_c")

using CSV, DataFrames, StatsPlots

# Read CSV file for one method
method1_df = CSV.read("ude_f1_scores.csv", DataFrame)

# Add a column to label this method
method1_df[!, :Method] .= "UDE-NN"

# Repeat this for method2 and method3 (assuming similar CSV files exist)
method2_df = CSV.read("gp_f1_scores.csv", DataFrame)
method2_df[!, :Method] .= "DE-GP"

method3_df = CSV.read("kernel_f1_scores.csv", DataFrame)
method3_df[!, :Method] .= "DE-KERNEL"

# Combine all DataFrames
combined_df = vcat(method1_df, method2_df, method3_df)

# Check the final combined data
# println(combined_df)
rename!(combined_df, :fscore_1 => :Regime_I, 
                        :fscore_2 => :Regime_II, 
                        :fscore_3 => :Regime_III)

# # Transform data for boxplot
df_long = stack(combined_df, [:Regime_I, :Regime_II, :Regime_III])  # Convert to long format
rename!(df_long, :variable => :FScore_Label, :value => :Score)


gr(tickfontsize=12,labelfontsize=14, legendfontsize=12)
# Define color mapping for methods
colors = Dict("DE-GP" => "#4477AA", "UDE-NN" => "#228833", "DE-KERNEL" => "#AA3377")

# Create grouped boxplot with custom settings
p = @df df_long groupedboxplot(:FScore_Label, :Score, group=:Method, 
                            # label=["DE-GP" "\\textbf{UDE-NN}" "DE-KERNEL"],
                               bar_width=0.7, legend=:bottomleft, dpi=400, 
                               color=[colors[m] for m in df_long.Method])

                            #    , label=["DE-GP" "UDE-NN" "DE-KERNEL"]
# Customizing legend box outline
plot!(p, framestyle=:box, 
         foreground_color_legend=nothing)
ylabel!("F-scores")
savefig("plots/grouped_fscores.png")