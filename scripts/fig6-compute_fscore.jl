using Pkg
Pkg.activate("C:/Users/kj_ra/Box/Ecology/Project_CStar/HPC_codes/discrete_dym/env_c")

using JLD2, Plots
using Random
using StatsBase
import StatisticalMeasures.ConfusionMatrices as CM
using DataFrames, CSV


# filename = "../outputs/ude/ude_predictions_800_1322.jld2"  
# data = JLD2.load(filename)
# size(data["predictions"].true_c), size(data["predictions"].pred_c )


function categorize_c(pred_c)
    # Categorizes the predicted c values into three categories based on thresholds Cstar1 and Cstar2.
    # Cstar1 and Cstar2 are determined based on mechanistic model's behavior or domain knowledge.
    Cstar1 = 1.788448
    Cstar2 = 2.604652
    new_c = []
    for c in pred_c
        if c<Cstar1
            push!(new_c, 1)
        elseif c>Cstar1 && c<Cstar2
            push!(new_c,2)
        else
            push!(new_c,3)
        end
    end
    return new_c
end


function get_confusion_mat(true_y, pred_y)
    cm = CM.confmat(categorize_c(true_y), categorize_c(pred_y), levels=[1,2,3])
    return CM.matrix(cm)
end
    

# pred_c = categorize_c(data["predictions"].pred_c)
# true_c = categorize_c(data["predictions"].true_c)
# cm = CM.confmat(true_c, pred_c, levels=[1,2,3])
# CM.matrix(cm)

function compute_f1_scores(confusion_matrix)
    # Number of classes
    n_classes = size(confusion_matrix, 1)
    
    # Initialize arrays for precision, recall, and F1-score for each class
    precision = zeros(Float64, n_classes)
    recall = zeros(Float64, n_classes)
    f1_scores = zeros(Float64, n_classes)
    
    # Total samples for weighted average
    total_support = sum(confusion_matrix)
    
    # Compute precision, recall, and F1-score for each class
    for i in 1:n_classes
        tp = confusion_matrix[i, i]                        # True positives
        fp = sum(confusion_matrix[:, i]) - tp              # False positives
        fn = sum(confusion_matrix[i, :]) - tp              # False negatives
        support = sum(confusion_matrix[i, :])              # Total true instances (support)
        
        # Precision and recall with handling for zero division
        precision[i] = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
        recall[i] = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
        f1_scores[i] = (precision[i] + recall[i]) > 0 ? 
                2 * precision[i] * recall[i] / (precision[i] + recall[i]) : 0.0
    end
    
    # Macro-average (simple average of F1-scores)
    macro_f1 = mean(f1_scores)
    
    # Weighted-average F1-score
    weights = sum(confusion_matrix, dims=2) ./ total_support
    weighted_f1 = sum(f1_scores .* weights)
    
    return f1_scores, macro_f1, weighted_f1
end


# Compute F1-scores
# f1_scores, macro_f1, weighted_f1 = compute_f1_scores(CM.matrix(cm))

# 

# for a single sample file
# filename = "outputs/kernel/kernel_predictions_800_16782.jld2"
# data = JLD2.load(filename)
# println("=============================================")
# println("Value Counts:")
# vc_true = countmap(categorize_c(data["predictions"].true_c))
# println(vc_true)
# vc_pred = countmap(categorize_c(data["predictions"].pred_c))
# println(vc_pred)
# println(minimum(data["predictions"].pred_c), "\t",
#             mean(data["predictions"].pred_c), "\t",
#             maximum(data["predictions"].pred_c))
# println("=============================================")
# cm = get_confusion_mat(data["predictions"].true_c, data["predictions"].pred_c)

function get_stats(x_vec)
    # Computes minimum, mean, and maximum of a vector.
    return minimum(x_vec), mean(x_vec), maximum(x_vec)
end


function eval_fscores(seeds, method)
    df = DataFrame(seed=String[], fscore_1=Float64[],
                    fscore_2=Float64[], fscore_3=Float64[],      
                        macro_f1=Float64[], weighted_f1=Float64[])
    file_none = []
    for seed in seeds
        # skip and posh filename to file if file does not exist
        filename = "outputs_latest/$(method)/$(method)_predictions_800_$(seed).jld2"
        try
            println("=============================================")
            data = JLD2.load(filename)  # Attempt to load the file
        
            println("Min, Mean, Max of true_c: ", get_stats(data["predictions"].true_c))
            println("Min, Mean, Max of pred_c: ", get_stats(data["predictions"].pred_c))

            vc_true = countmap(categorize_c(data["predictions"].true_c))
            println(vc_true)
            vc_pred = countmap(categorize_c(data["predictions"].pred_c))
            println(vc_pred)
            cm = get_confusion_mat(data["predictions"].true_c, data["predictions"].pred_c)
            println(cm)
            println()
            f1_scores, macro_f1, weighted_f1 = compute_f1_scores(cm)
            println("F1: ", f1_scores)
            println("Macro F1: ", macro_f1, " Weighted F1: ", weighted_f1)   
            push!(df, (seed, f1_scores[1],f1_scores[2], f1_scores[3],
                                macro_f1, weighted_f1))
        catch e
            println("An error occurred: ", e)
            push!(file_none, filename)  # Store the filename that caused the error
            continue  # Skip to the next iteration
        end
    end
    # return C_CMS, mean(F_SCORES)#, mean(ACCURACIES)
    # save df to csv
    CSV.write("$(method)_f1_scores.csv", df)
    println("F1 scores saved to $(method)_f1_scores.csv")

    println("Files that could not be loaded:")
    for file in file_none
        println(file)
    end
end


f = open("rnds.txt", "r")
seeds = []
for lines in readlines(f)
    push!(seeds, lines)
end

eval_fscores(seeds, "ude")
println("=============================================")
eval_fscores(seeds, "kernel")   
println("=============================================")
eval_fscores(seeds, "gp")   
