using JuMP, CPLEX
using DataFrames
using Distributions
using Random
using LinearAlgebra

H = 12
J = 6
T = 10
s = fill(0, H, J)

for h in 1:6
    for j in 1:6
        if h == j
            s[h,j] = 1
        end
    end
end

for h in 7:12
    for j in 1:6
        if (h - 6) == j
            s[h,j] = 1
        end
    end
end

s

zᵖ = [7;
      7;
      9;
      9;
      11;
      11;
      7;
        7;
        9;
        9;
        11;
        11
] * 1000

zᵐ = fill(0, H, J)

for h in 1:4
    for j in 1:6
        zᵐ[h , h+1] = (1/2) * zᵖ[h]
        zᵐ[h , h+2] = (3/4) * zᵖ[h]
    end
end

for h in 7:10
    for j in 1:6
        zᵐ[h , h-6+1] = (1/2) * zᵖ[h]
        zᵐ[h , h-6+2] = (3/4) * zᵖ[h]
    end
end

zᵐ

μ = fill(0.5, J)

Σ = Diagonal(fill(0.1, J))

Random.seed!(123)
d = round.(rand(MvNormal(μ , Σ), T ), digits = 1)

for j in 1:J
    for t in 1:T
        if d[j,t] < 0
            d[j,t] = 1
        end
    end
end

d = [
    0.7  0.8  0.8  0.6  0.9  1.0  1.0  0.9  0.0  1.0
    0.2  0.2  1.0  0.9  0.7  0.7  0.4  1.0  0.3  0.7
    0.5  0.6  0.5  0.1  0.4  0.6  0.4  0.0  0.5  0.6
    0.5  1.4  0.3  1.0  0.8  0.8  0.2  0.8  0.6  0.6
    0.8  0.2  1.3  0.6  0.3  0.4  0.6  0.8  0.4  0.8
    0.3  0.4  0.9  0.2  0.6  1.0  0.4  0.3  0.4  0.6
]



m = Model(CPLEX.Optimizer)
@variable(m, ψ[1:H , 1:J], Bin)
@variable(m, α[h in 1:H , j in 1:J , t in 1:T], Bin)


@objective(m, Min, sum(ψ[h,j] * zᵖ[h] * T for h in 1:H for j in 1:J ) )

@constraint(m, c1[h in 1:H , j in 1:J], ψ[h,j] ≤ s[h,j])

@constraint(m, c2[j in 1:J , t in 1:T], sum( α[h,j,t] for h in 1:H ) ≥ d[j,t] )
@constraint(m, c3[h in 1:H , j in 1:J , t in 1:T], α[h,j,t] ≤ ψ[h,j])
@constraint(m, c4[h in 1:H  , t in 1:T], sum( α[h,j,t] for j in 1:J) ≤ 1 )

optimize!(m)

@show objective_value(m)

value(sum(ψ))
df_ψ = DataFrame(value.(ψ), :auto)
rename!(df_ψ,[:j1,:j2, :j3])

df_α1 = DataFrame( value.(α[: , : , 1]), :auto)
rename!(df_α1, [:j1,:j2, :j3])

df_α2 = DataFrame(value.(α[: , : , 2]), :auto)
rename!(df_α2, [:j1,:j2, :j3])

@show df_ψ
@show df_α1
@show df_α2


println("the total number of recruited people are : ", value(sum(ψ)))

for j in 1:J
    ζ =  value(sum(ψ[h,j] for h in 1:H) )
    return ζ
end
ζ
value(sum(ψ[h,1] for h in 1:H) )
value(sum(ψ[h,2] for h in 1:H) )
value(sum(ψ[h,3] for h in 1:H) )
value(sum(ψ[h,4] for h in 1:H) )
value(sum(ψ[h,5] for h in 1:H) )
value(sum(ψ[h,6] for h in 1:H) )
