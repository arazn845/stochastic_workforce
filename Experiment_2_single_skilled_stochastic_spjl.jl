using StochasticPrograms
using CPLEX
using Distributions
using Plots
using DataFrames
using Random
using LinearAlgebra

###################################
###################################

Ξ = 1
H = 12
J = 6
Tᵈ = 5
Tˢ = 5
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

zᶜ = zᵖ *2

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

zᵐ[5,6] = 11000 * 1/2
zᵐ[5,1] = 11000 * 3/4

zᵐ[6,1] = 11000 * 1/2
zᵐ[6,2] = 11000 * 3/4

zᵐ[11,6] = 11000 * 1/2
zᵐ[11,1] = 11000 * 3/4

zᵐ[12,1] = 11000 * 1/2
zᵐ[12,2] = 11000 * 3/4

zᵐ



d = [
    0.7  0.8  0.8  0.6  0.9  1.0  1.0  0.9  0.0  1.0
    0.2  0.2  1.0  0.9  0.7  0.7  0.4  1.0  0.3  0.7
    0.5  0.6  0.5  0.1  0.4  0.6  0.4  0.0  0.5  0.6
    0.5  1.4  0.3  1.0  0.8  0.8  0.2  0.8  0.6  0.6
    0.8  0.2  1.3  0.6  0.3  0.4  0.6  0.8  0.4  0.8
    0.3  0.4  0.9  0.2  0.6  1.0  0.4  0.3  0.4  0.6
]

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

@sampler SimpleSampler = begin
    N::MvNormal
    SimpleSampler(µ, Σ) = new(MvNormal(µ, Σ))
        @sample Scenario begin
        x = rand(sampler.N , Tˢ)
        return @scenario dξ = round.(x, digits = 1)
    end
end

μ = fill(0.5 , J)

Σ = Diagonal(fill(0.1 , J))

#########################################################
#########################################################

sp = instantiate(skill, SimpleSampler(µ, Σ), Ξ)

set_optimizer(sp, CPLEX.Optimizer)

optimize!(sp)

@show objective_value(sp)
@show DataFrame( value.(sp[1, :ψ]), :auto ) 
@show DataFrame( value.(sp[1, :ψ]), :auto ) 



value.(sp[1, :α])

sp[2 , :γξ]

sum(value.(sp[2 , :γξ], i) for i in 1:10 ) .* (1/ Ξ)


