using JuMP
using CPLEX
using DataFrames
using Distributions
using Plots
using CSV


H = 4
J = 3
Tᵈ = 1
Tˢ = 1
Ξ = 1

μ = [1.2, 0.8, 0.6]

Σ = [0.5 0.0 0.0;
     0.0 0.5 0.0;
     0.0 0.0 0.5
    ]

dξ = reshape(rand(MvNormal(μ,Σ), Ξ ), (J,Tˢ,Ξ) )

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
] * 1000;


main = Model(CPLEX.Optimizer)
@variable(main, 0 ≤ θ)
@variable(main, ψ[1:H , 1:J], Bin)
@variable(main, 0 ≤ χ[1:H , 1:J] ≤ 1)
@variable(main, α[1:H , 1:J , 1:Tᵈ], Bin)
@variable(main, z1[1:H , 1:J , 1:Tᵈ], Bin)
@variable(main, z2[1:H , 1:J , 1:Tᵈ], Bin)

@objective(main, Min, sum( zᵖ[h] * ψ[h,j] * (Tᵈ + Tˢ) for h in 1:H for j in 1:J ) +
                    sum(zᵐ[h,j] * χ[h,j] for h in 1:H for j in 1:J ) )
    
@constraint(main, linear1[h = 1:H , j = 1:J , t =1:Tᵈ], z1[h,j,t] ≤ α[h,j,t] )
@constraint(main, linear2[h = 1:H , j = 1:J , t =1:Tᵈ], z1[h,j,t] ≤ ψ[h,j] )
@constraint(main, linear3[h = 1:H , j = 1:J , t =1:Tᵈ], z1[h,j,t] ≥ α[h,j,t] + ψ[h,j] -1 )

@constraint(main, linear4[h = 1:H , j = 1:J , t =1:Tᵈ], z2[h,j,t] ≤ α[h,j,t] )
@constraint(main, linear5[h = 1:H , j = 1:J , t =1:Tᵈ], z2[h,j,t] ≤ χ[h,j] )
@constraint(main, linear6[h = 1:H , j = 1:J , t =1:Tᵈ], z2[h,j,t] ≥ α[h,j,t] + χ[h,j] -1 )

@constraint(main, demand_met[ j in 1:J , t in 1:Tᵈ ], sum(z1[h,j,t] + z2[h,j,t] for h in 1:H ) ≥ d[j,t] )
@constraint(main, skill_in_recruitment[h in 1:H , j in 1:J], ψ[h,j] ≤ s[h,j] );
@constraint(main, skill_in_training[h in 1:H , j in 1:J], χ[h,j] ≤ 1 - ψ[h,j] );
@constraint(main, train_only_recruited[ h in 1:H ], sum(χ[h,j] for j in 1:J ) ≤ sum(ψ[h,j] for j in 1:J) * J  );
@constraint(main, no_more_than_one_station[ h in 1:H , t in 1:Tᵈ ], sum(α[h,j,t] for j in 1:J) ≤ 1 );
@constraint(main, permanent_allocability[h in 1:H , j in 1:J , t in 1:Tᵈ], α[h,j,t] ≤ ψ[h,j] + 10 * χ[h,j] )



#####################################################
# subproblem
#####################################################
ψ = rand(H,J)
χ = rand(H,J)
sub = Model(CPLEX.Optimizer)
@variable(sub, 0 ≤ γξ[1:J , 1:Tˢ, 1:Ξ])
@variable(sub, αξ[1:H , 1:J , 1:Tˢ, 1:Ξ], Bin)
@variable(sub, z3[1:H , 1:J , 1:Tˢ, 1:Ξ], Bin)
@variable(sub, z4[1:H , 1:J , 1:Tˢ, 1:Ξ])

@objective(sub, Min, sum(zᶜ[j] * γξ[j,t,ξ] for j in 1:J for t in 1:Tˢ for ξ in 1:Ξ ) )

@constraint(sub, linear7[h = 1:H , j = 1:J , t =1:Tˢ, ξ = 1:Ξ], z3[h,j,t,ξ] ≤ αξ[h,j,t,ξ] )
@constraint(sub, linear8[h = 1:H , j = 1:J , t =1:Tˢ, ξ = 1:Ξ], z3[h,j,t,ξ] ≤ ψ[h,j] )
@constraint(sub, linear9[h = 1:H , j = 1:J , t =1:Tˢ, ξ =1:Ξ], z3[h,j,t,ξ] ≥ αξ[h,j,t,ξ] + ψ[h,j] -1 )

@constraint(sub, linear10[h = 1:H , j = 1:J , t =1:Tˢ, ξ =1:Ξ], z4[h,j,t,ξ] ≤ αξ[h,j,t,ξ] )
@constraint(sub, linear11[h = 1:H , j = 1:J , t =1:Tˢ, ξ =1:Ξ], z4[h,j,t,ξ] ≤ χ[h,j] )
@constraint(sub, linear12[h = 1:H , j = 1:J , t =1:Tˢ, ξ =1:Ξ], z4[h,j,t,ξ] ≥ αξ[h,j,t,ξ] + χ[h,j] -1 )

con1 = @constraint(sub, stage2_demand_met[j = 1:J , t =1:Tˢ, ξ=1:Ξ], sum(z3[h,j,t,ξ] for h in 1:H ) + sum(z4[h,j,t,ξ] for h in 1:H) + γξ[j,t,ξ] ≥ dξ[j,t,ξ] )
con2 = @constraint(sub, stage2_permanent_allocability[h in 1:H , j in 1:J , t in 1:Tˢ, ξ in 1:Ξ], αξ[h,j,t,ξ] ≤ ψ[h,j] + 10 * χ[h,j] )
con3 = @constraint(sub, stage2_no_more_than_one_station[ h in 1:H , t in 1:Tˢ , ξ in 1:Ξ ], sum(αξ[h,j,t,ξ] for j in 1:J) ≤ 1 )

optimize!(sub)

dual.(con1)
