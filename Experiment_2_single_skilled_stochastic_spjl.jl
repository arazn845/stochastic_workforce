using StochasticPrograms
using GLPK
using CPLEX
using Distributions
using Plots
using LinearAlgebra

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

χᵐᵃˣ = fill(1, H, J)


@stochastic_model skill begin
   @stage 1 begin
        @decision(skill, ψ[1:H , 1:J], Bin)
        @decision(skill, 0 ≤ χ[1:H , 1:J])
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
        @constraint(skill, permanent_pool[h in 1:H , j in 1:J], ψ[h,j] ≤ s[h,j] );
        @constraint(skill, allocation_recruitment_training[h in 1:H , j in 1:J , t in 1:Tᵈ], α[h,j,t] ≤ ψ[h,j] + 10 * χ[h,j] )
        @constraint(skill, no_more_than_one_station[ h in 1:H , t in 1:Tᵈ ], sum(α[h,j,t] for j in 1:J) ≤ 1 );
        @constraint(skill, single_skilling_at_recruitment[h in 1:H , j in 1:J], χ[h,j] ≤ ( 1 - ψ[h,j] ) * χᵐᵃˣ[h,j] );
        @constraint(skill, training_recruited[ h in 1:H ], sum(χ[h,j] for j in 1:J ) ≤ sum(ψ[h,j] for j in 1:J) * J  );
        
        
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
        
        @constraint(skill, stage2_demand_met[j = 1:J , t =1:Tˢ], sum(z3[h,j,t] for h in 1:H ) + sum(z4[h,j,t] for h in 1:H) + γξ[j,t] ≥ dξ[j,t] )
        @constraint(skill, stage2_permanent_allocability[h in 1:H , j in 1:J , t in 1:Tˢ], αξ[h,j,t] ≤ ψ[h,j] + 10 * χ[h,j] )
        @constraint(skill, stage2_no_more_than_one_station[ h in 1:H , t in 1:Tˢ ], sum(αξ[h,j,t] for j in 1:J) ≤ 1 )
    end
end


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

#######


sp = instantiate(skill, SimpleSampler(µ, Σ), Ξ)

set_optimizer(sp, CPLEX.Optimizer)

optimize!(sp)

