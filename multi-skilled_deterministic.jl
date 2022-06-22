using JuMP
using CPLEX
using DataFrames

#############################################
#############################################
H = 4
J = 3
Tᵈ = 2

s = [
    1 0 0;
    1 0 0; 
    0 1 0;
    0 0 1
] 

d = [
    0.3  1.2   
    0.2  0.2
    1.0  0.0
]

zᵖ = [7;
      7;
      9;
      11
] * 1000

zᶜ = zᵖ * 2

zᵐ = [
    0 3 5;
    0 3 5;
    1 0 3;
    2 4 0
] * 1000
#############################################
#############################################

m = Model(CPLEX.Optimizer);

@variable(m, ψ[1:H , 1:J], Bin)
@variable(m, 0 ≤ χ[1:H , 1:J] ≤ 1)
@variable(m, α[1:H , 1:J , 1:Tᵈ], Bin)
@variable(m, z1[1:H , 1:J , 1:Tᵈ], Bin)
@variable(m, z2[1:H , 1:J , 1:Tᵈ], Bin)

@objective(m, Min, sum( zᵖ[h] * ψ[h,j] * Tᵈ for h in 1:H for j in 1:J ) +
                    sum(zᵐ[h,j] * χ[h,j] for h in 1:H for j in 1:J ) 
)

@constraint(m, linear1[h = 1:H , j = 1:J , t =1:Tᵈ], z1[h,j,t] ≤ α[h,j,t] )
@constraint(m, linear2[h = 1:H , j = 1:J , t =1:Tᵈ], z1[h,j,t] ≤ ψ[h,j] )
@constraint(m, linear3[h = 1:H , j = 1:J , t =1:Tᵈ], z1[h,j,t] ≥ α[h,j,t] + ψ[h,j] -1 )

@constraint(m, linear4[h = 1:H , j = 1:J , t =1:Tᵈ], z2[h,j,t] ≤ α[h,j,t] )
@constraint(m, linear5[h = 1:H , j = 1:J , t =1:Tᵈ], z2[h,j,t] ≤ χ[h,j] )
@constraint(m, linear6[h = 1:H , j = 1:J , t =1:Tᵈ], z2[h,j,t] ≥ α[h,j,t] + χ[h,j] -1 )

@constraint(m, demand_met[ j in 1:J , t in 1:Tᵈ ], sum(z1[h,j,t] + z2[h,j,t] for h in 1:H ) ≥ d[j,t] )

@constraint(m, c1[h in 1:H , j in 1:J], ψ[h,j] ≤ s[h,j] );

@constraint(m, c2[h in 1:H , j in 1:J], χ[h,j] ≤ 1 - ψ[h,j] );

@constraint(m, c3[ h in 1:H ], sum(χ[h,j] for j in 1:J ) ≤ sum(ψ[h,j] for j in 1:J) * J  );

@constraint(m, c4[ h in 1:H , t in 1:Tᵈ ], sum(α[h,j,t] for j in 1:J) ≤ 1 );

@constraint(m, c6[h in 1:H , j in 1:J , t in 1:Tᵈ], α[h,j,t] ≤ ψ[h,j] + 10 * χ[h,j] )

optimize!(m)
@show objective_value(m)


df_ψ = DataFrame(value.(ψ), :auto)
rename!(df_ψ,[:j1,:j2, :j3])

df_χ = DataFrame(value.(χ), :auto)
rename!(df_χ,[:j1,:j2, :j3])

df_α1 = DataFrame( value.(α[: , : , 1]), :auto)
rename!(df_α1, [:j1,:j2, :j3])

df_α2 = DataFrame(value.(α[: , : , 2]), :auto)
rename!(df_α2, [:j1,:j2, :j3])

println("permanent workforce cost = " , value(sum( zᵖ[h] * ψ[h,j] * Tᵈ for h in 1:H for j in 1:J ) ))

println("Training cost = " , value( sum(zᵐ[h,j] * χ[h,j] for h in 1:H for j in 1:J ) ))

@show df_ψ
@show df_χ
@show df_α1
@show df_α2
