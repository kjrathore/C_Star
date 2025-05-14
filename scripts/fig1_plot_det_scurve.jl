# lets prepare s curve diagram for Tilman's data model.
# https://github.com/JuliaMath/Roots.jl
using Pkg
Pkg.activate("env")

# Suggeted By GPT:
using Roots, Plots
using Polynomials;
using LaTeXStrings

r = 1.0
K = 10.0
h = 1.0
# c = 3.1

# p = Polynomial([0, r*h^2, -(r*h^2/K + c), r, -r/K])  
# rts = roots(p)
# println(rts)

eqs = zeros(1001,4)
c_vals = collect(range(0.0, 4.0, length=1001))
# FIND EQUILIBRIA
for (i, c_val) in enumerate(c_vals)
    global c
    c = c_val
    # p = Polynomial([-r/K, r, -(r*h^2/K + c), r*h^2, 0])#[-1/K, 1, (h^2/K-c/r), h^2, 0])  # e.g., z^3 - 1
    p = Polynomial([0, r*h^2,-(r*h^2/K + c), r, -r/K])
    rts = roots(p)
    # println("c:",c, "\t roots:", abs.(imag.(rts)))
    # eqs[i,:] = real(rts)
    im_vals = abs.(imag.(rts))
    for (ind, rt) in enumerate(rts)
        if im_vals[ind] > .025
            rts[ind] = NaN
        end
    end
    eqs[i,:] = real(rts)

end

#max function with nan
arg_max(x) = findmax(x -> isfinite(x) ? x : -Inf, x)[2]
arg_min(x) = findmin(x -> isfinite(x) ? x : Inf, x)[2]
#collect regime boudaries
Cstar1 = c_vals[arg_min(eqs[:,2])]
Cstar2 = c_vals[arg_max(eqs[:,2])]  
# for i in 1:1000
#     println(i, " ", eqs[i,:])
# end

println()
println("Cstar1 t  ",arg_min(eqs[:,2]))
println("Cstar2 t ",arg_max(eqs[:,2]))
println("Cstar1  ",Cstar1, arg_min(eqs[:,2]))
println("Cstar2  ",Cstar2, arg_max(eqs[:,2]))


gr(legendfontsize=12, markerstrokewidth =0, 
        xtickfontsize=10, ytickfontsize=10, 
        xguidefontsize=12, yguidefontsize=12,
        dpi=600)

plot(c_vals, eqs[:,1], lw=4, color=:orange, label="")
plot!(c_vals, eqs[:,3], lw=4, color=:orange, label="Stable")
plot!(c_vals, eqs[:,2], label="Unstable", lw=4, ls=:dash, color=:black)
plot!(c_vals, eqs[:,4], label="", lw=4, ls=:dash, color=:black)
annotate!(0.8,5.0, ("Regime I", 12))
annotate!(2.2,5.0, ("Regime II", 12))
annotate!(3.2,5.0, ("Regime III", 12))
s = plot!([Cstar1, Cstar2], seriestype=:vline, 
        lw=3, color=:black, alpha=0.4, label=L"c*",
        xlabel=L"extraction rate, $c$",
        ylabel=L"environmental state, $x$")

png(s,"plots/scurve.png")

# # Algebric formulation
# # 0   = r*x*(1-x/K)-c*(x^2/(x^2+h^2))
# #     = (rx-rx^2/K)*(x^2-h^2) - cx^2
# #     = rx^3 - rh^2x - rx^4/K + rh^2x^2/K - cx^2
# #     = -rx^4/K + rx^3 + (rh^2/K - c)x^2 - rh^2x

# #     [-r/K, r, (rh^2/K - c), -rh^2, 0]  
# = [-1/K, 1, (h^2/K-c/r), -h^2, 0]  #need to reverse this for julia Polynomials.