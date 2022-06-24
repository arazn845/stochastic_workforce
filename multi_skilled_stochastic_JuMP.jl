using JuMP
using CPLEX
using DataFrames
using Distributions
using Plots
using CSV

###############################
###############################

μ = [1.2, 0.8, 0.6]

Σ = [0.5 0.0 0.0;
     0.0 0.5 0.0;
     0.0 0.0 0.5
    ]



#############################################
#############################################
H = 4
J = 3
Tᵈ = 1
Tˢ = 1
Ξ = 500

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
] * 1000
#############################################
#############################################
skill = Model(CPLEX.Optimizer)
#####################################################
# stage 1 variables
#####################################################
@variable(skill, ψ[1:H , 1:J], Bin)
@variable(skill, 0 ≤ χ[1:H , 1:J] ≤ 1)
@variable(skill, α[1:H , 1:J , 1:Tᵈ], Bin)
@variable(skill, z1[1:H , 1:J , 1:Tᵈ], Bin)
@variable(skill, z2[1:H , 1:J , 1:Tᵈ], Bin)
#####################################################
# stage 2 variables
#####################################################
@variable(skill, 0 ≤ γξ[1:J , 1:Tˢ, 1:Ξ])
@variable(skill, αξ[1:H , 1:J , 1:Tˢ, 1:Ξ], Bin)
@variable(skill, z3[1:H , 1:J , 1:Tˢ, 1:Ξ], Bin)
@variable(skill, z4[1:H , 1:J , 1:Tˢ, 1:Ξ], Bin)

@objective(skill, Min, sum( zᵖ[h] * ψ[h,j] * (Tᵈ + Tˢ) for h in 1:H for j in 1:J ) +
                    sum(zᵐ[h,j] * χ[h,j] for h in 1:H for j in 1:J ) +
                    (1/Ξ) * sum(zᶜ[j] * γξ[j,t,ξ] for j in 1:J for t in 1:Tˢ for ξ in 1:Ξ )
)



@constraint(skill, linear1[h = 1:H , j = 1:J , t =1:Tᵈ], z1[h,j,t] ≤ α[h,j,t] )
@constraint(skill, linear2[h = 1:H , j = 1:J , t =1:Tᵈ], z1[h,j,t] ≤ ψ[h,j] )
@constraint(skill, linear3[h = 1:H , j = 1:J , t =1:Tᵈ], z1[h,j,t] ≥ α[h,j,t] + ψ[h,j] -1 )

@constraint(skill, linear4[h = 1:H , j = 1:J , t =1:Tᵈ], z2[h,j,t] ≤ α[h,j,t] )
@constraint(skill, linear5[h = 1:H , j = 1:J , t =1:Tᵈ], z2[h,j,t] ≤ χ[h,j] )
@constraint(skill, linear6[h = 1:H , j = 1:J , t =1:Tᵈ], z2[h,j,t] ≥ α[h,j,t] + χ[h,j] -1 )

@constraint(skill, demand_met[ j in 1:J , t in 1:Tᵈ ], sum(z1[h,j,t] + z2[h,j,t] for h in 1:H ) ≥ d[j,t] )
@constraint(skill, skill_in_recruitment[h in 1:H , j in 1:J], ψ[h,j] ≤ s[h,j] );
@constraint(skill, skill_in_training[h in 1:H , j in 1:J], χ[h,j] ≤ 1 - ψ[h,j] );
@constraint(skill, train_only_recruited[ h in 1:H ], sum(χ[h,j] for j in 1:J ) ≤ sum(ψ[h,j] for j in 1:J) * J  );
@constraint(skill, no_more_than_one_station[ h in 1:H , t in 1:Tᵈ ], sum(α[h,j,t] for j in 1:J) ≤ 1 );
@constraint(skill, permanent_allocability[h in 1:H , j in 1:J , t in 1:Tᵈ], α[h,j,t] ≤ ψ[h,j] + 10 * χ[h,j] )

###############################################################
# Stage 2 constraints
###############################################################

@constraint(skill, linear7[h = 1:H , j = 1:J , t =1:Tˢ, ξ = 1:Ξ], z3[h,j,t,ξ] ≤ αξ[h,j,t,ξ] )
@constraint(skill, linear8[h = 1:H , j = 1:J , t =1:Tˢ, ξ = 1:Ξ], z3[h,j,t,ξ] ≤ ψ[h,j] )
@constraint(skill, linear9[h = 1:H , j = 1:J , t =1:Tˢ, ξ =1:Ξ], z3[h,j,t,ξ] ≥ αξ[h,j,t,ξ] + ψ[h,j] -1 )

@constraint(skill, linear10[h = 1:H , j = 1:J , t =1:Tˢ, ξ =1:Ξ], z4[h,j,t,ξ] ≤ αξ[h,j,t,ξ] )
@constraint(skill, linear11[h = 1:H , j = 1:J , t =1:Tˢ, ξ =1:Ξ], z4[h,j,t,ξ] ≤ χ[h,j] )
@constraint(skill, linear12[h = 1:H , j = 1:J , t =1:Tˢ, ξ =1:Ξ], z4[h,j,t,ξ] ≥ αξ[h,j,t,ξ] + χ[h,j] -1 )

@constraint(skill, stage2_demand_met[j = 1:J , t =1:Tˢ, ξ=1:Ξ], sum(z3[h,j,t,ξ] for h in 1:H ) + sum(z4[h,j,t,ξ] for h in 1:H) + γξ[j,t,ξ] ≥ dξ[j,t,ξ] )
@constraint(skill, stage2_permanent_allocability[h in 1:H , j in 1:J , t in 1:Tˢ, ξ in 1:Ξ], αξ[h,j,t,ξ] ≤ ψ[h,j] + 10 * χ[h,j] )
@constraint(skill, stage2_no_more_than_one_station[ h in 1:H , t in 1:Tˢ , ξ in 1:Ξ ], sum(αξ[h,j,t,ξ] for j in 1:J) ≤ 1 )

optimize!(skill)

println("objective value = ", objective_value(skill) )

println("permanent workforce cost = " , value(sum( zᵖ[h] * ψ[h,j] * (Tᵈ + Tˢ) for h in 1:H for j in 1:J ) ))

println("Training cost = " , value( sum(zᵐ[h,j] * χ[h,j] for h in 1:H for j in 1:J ) ))

println("Average casual workforce cost = ", value( (1/Ξ) * sum(zᶜ[j] * γξ[j,t,ξ] for j in 1:J for t in 1:Tˢ for ξ in 1:Ξ ) )  )

df_ψ = DataFrame(value.(ψ), :auto)
rename!(df_ψ,[:j1,:j2, :j3])

df_χ = DataFrame(value.(χ), :auto)
rename!(df_χ,[:j1,:j2, :j3])

df_α1 = DataFrame( value.(α[: , : , 1]), :auto)
rename!(df_α1, [:j1,:j2, :j3])

@show df_ψ
@show df_χ
@show df_α1

DF_γξ = DataFrame(value.(γξ[: , : , 1]), :auto)


a = DataFrame(value.(γξ)[1, : , : ], :auto)

b = DataFrame(value.(γξ)[2, : , : ], :auto)

c = DataFrame(value.(γξ)[2, : , : ], :auto)

pwd()

CSV.write("a.csv", a)
CSV.write("b.csv", b)
CSV.write("c.csv", c)


dξ[1, : , :]

skill_1 = DataFrame(dξ[1, : , :], :auto)
skill_2 = DataFrame(dξ[2, : , :], :auto)
skill_3 = DataFrame(dξ[3, : , :], :auto)
 
CSV.write("skill_1.csv", skill_1)
CSV.write("skill_2.csv", skill_2)
CSV.write("skill_3.csv", skill_3)

