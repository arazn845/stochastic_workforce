using StochasticPrograms, CPLEX
using Distributions
using Plots
###############################################
###############################################
H = 4
J = 3
Tᵈ = 1
Tˢ = 1
Ξ = 650

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
#######################################################
######################################################

skill = @stochastic_model begin

    @stage 1 begin
        @decision(model, ψ[1:H , 1:J], Bin)
        @decision(model, α[h in 1:H , j in 1:J , t in 1:Tᵈ], Bin) 
        
        @objective(model, Min, sum(ψ[h,j] * zᵖ[h] * (Tˢ + Tᵈ) for h in 1:H for j in 1:J ) )

        @constraint(model, s1_c1[h in 1:H , j in 1:J], ψ[h,j] ≤ s[h,j])
        @constraint(model, s1_c2[j in 1:J , t in 1:Tᵈ], sum( α[h,j,t] for h in 1:H ) ≥ d[j,t] )
        @constraint(model, s1_c3[h in 1:H , j in 1:J , t in 1:Tᵈ], α[h,j,t] ≤ ψ[h,j])
        @constraint(model, s1_c4[h in 1:H , t in 1:Tᵈ], sum( α[h,j,t] for j in 1:J) ≤ 1 )
    end
    
    @stage 2 begin
        @uncertain dξ
        @recourse(model, αξ[1:H , 1:J , 1:Tˢ], Bin)
        @recourse(model, 0 ≤ γξ[1:J , 1:Tˢ])
        
        @objective(model, Min, sum( zᶜ[j] * γξ[j,t] for j in 1:J for t in 1:Tˢ ) )
        
        @constraint(model, s2_c1[j in 1:J , t in 1:Tˢ], sum(αξ[h,j,t] for h in 1:H) + γξ[j,t] ≥ dξ[j,t])
        @constraint(model, s2_c2[h in 1:H , j in 1:J , t in 1:Tˢ], αξ[h,j,t] ≤ ψ[h,j] )
        @constraint(model, s2_c3[h in 1:H , t in 1:Tˢ], sum(αξ[h,j,t] for j in 1:J) ≤ 1 )
    end
    
end
#########################################################
#########################################################
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
#########################################################
#########################################################
sp = instantiate(skill, SimpleSampler(µ, Σ), Ξ)

set_optimizer(sp, CPLEX.Optimizer)

optimize!(sp)

value.(sp[1, :ψ])

value.(sp[1, :α])

sp[2 , :γξ]

sum(value.(sp[2 , :γξ], i) for i in 1:10 ) .* (1/ Ξ)


objective_value(sp)

