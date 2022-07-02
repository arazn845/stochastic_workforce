using StochasticPrograms
using CPLEX
using DataFrames
using Distributions

###############################
###############################


#############################################
#############################################
H = 4
J = 3
Tᵈ = 1
Tˢ = 1

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
@stochastic_model skill begin
   @stage 1 begin
        @decision(skill, ψ[1:H , 1:J], Bin)
        @decision(skill, 0 ≤ χ[1:H , 1:J] ≤ 1)
        @decision(skill, α[1:H , 1:J , 1:Tᵈ], Bin)
        @decision(skill, z1[1:H , 1:J , 1:Tᵈ], Bin)
        @decision(skill, z2[1:H , 1:J , 1:Tᵈ])
        
        @objective(skill, Min, sum( zᵖ[h] * ψ[h,j] * (Tᵈ + Tˢ) for h in 1:H for j in 1:J ) +
                            sum(zᵐ[h,j] * χ[h,j] for h in 1:H for j in 1:J ) 
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
    end
    
    @stage 2 begin 
        
        @uncertain dξ
        @recourse(skill, 0 ≤ γξ[1:J , 1:Tˢ])
        @recourse(skill, αξ[1:H , 1:J , 1:Tˢ], Bin)
        @recourse(skill, z3[1:H , 1:J , 1:Tˢ], Bin)
        @recourse(skill, z4[1:H , 1:J , 1:Tˢ])

        @objective(skill, Min, sum(zᶜ[j] * γξ[j,t] for j in 1:J for t in 1:Tˢ ) )
        
        @constraint(skill, linear7[h = 1:H , j = 1:J , t =1:Tˢ], z3[h,j,t] ≤ αξ[h,j,t] )
        @constraint(skill, linear8[h = 1:H , j = 1:J , t =1:Tˢ], z3[h,j,t] ≤ ψ[h,j] )
        @constraint(skill, linear9[h = 1:H , j = 1:J , t =1:Tˢ], z3[h,j,t] ≥ αξ[h,j,t] + ψ[h,j] -1 )

        @constraint(skill, linear10[h = 1:H , j = 1:J , t =1:Tˢ], z4[h,j,t] ≤ αξ[h,j,t] )
        @constraint(skill, linear11[h = 1:H , j = 1:J , t =1:Tˢ], z4[h,j,t] ≤ χ[h,j] )
        @constraint(skill, linear12[h = 1:H , j = 1:J , t =1:Tˢ], z4[h,j,t] ≥ αξ[h,j,t] + χ[h,j] -1 )
        
        @constraint(skill, stage2_demand_met[j = 1:J , t =1:Tˢ], sum(z3[h,j,t] for h in 1:H ) + sum(z4[h,j,t] for h in 1:H) + γξ[j,t] ≥ dξ )
        @constraint(skill, stage2_permanent_allocability[h in 1:H , j in 1:J , t in 1:Tˢ], αξ[h,j,t] ≤ ψ[h,j] + 10 * χ[h,j] )
        @constraint(skill, stage2_no_more_than_one_station[ h in 1:H , t in 1:Tˢ ], sum(αξ[h,j,t] for j in 1:J) ≤ 1 )
    end
end

μ = [1.2, 0.8, 0.6]

Σ = [0.5 0.0 0.0;
     0.0 0.5 0.0;
     0.0 0.0 0.5
    ]
@sampler SimpleSampler = begin
    N::MvNormal
    SimpleSampler(µ, Σ) = new(MvNormal(µ, Σ))
        @sample Scenario begin
        x = rand(sampler.N)
        return @scenario dξ = round.(x, digits = 1)
    end
end

sp = instantiate(skill, SimpleSampler(µ, Σ), 1)

set_optimizer(sp, CPLEX.Optimizer)
