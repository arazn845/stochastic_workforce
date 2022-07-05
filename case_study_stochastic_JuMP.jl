using JuMP
using GLPK
using Gurobi
using Distributions
using Random
using Plots
using LinearAlgebra
using DataFrames


skill = Model(Gurobi.Optimizer)

Ξ = 10

H = 48
J = 12

Tᵈ = 3
Tˢ = 2

d = [
    2.5 2.5 2.5; #Labouring
    2.7 2.7 2.7 ; #Caulkering (bottleneck)
    2.2 2.2 2.2 ; #Mechanical controlling
    2.3 2.3 2.3 ; #Tiling
    2.5 2.5 2.5 ; #Plumbing
    2.5 2.5 2.5 ; #plastering
    2.7 2.7 2.7 ; #carpentery
    2.9 2.9 2.9 ; #electric work
    3.0 3.0 3.0 ; #water-proofinh
    2.8 2.8 2.8 ; #glazing
    2.7 2.7 2.7 ; #joining
    2.0 2.0 2.0 ; #painting
]


Random.seed!(123)
x1 = rand(Uniform(2.5,3.5), Tˢ, Ξ )
x2 = rand(Uniform(2.7,4.2), Tˢ, Ξ )
x3 = rand(Uniform(2.2,3.2),  Tˢ, Ξ  )
x4 = rand(Uniform(2.3,3.3),  Tˢ, Ξ  )
x5 = rand(Uniform(2.5,3.5),  Tˢ, Ξ  )
x6 = rand(Uniform(2.5,3.5),  Tˢ, Ξ  )
x7 = rand(Uniform(2.7,3.7),  Tˢ, Ξ  )
x8 = rand(Uniform(2.9,3.9),  Tˢ, Ξ  )
x9 = rand(Uniform(3.0,4.0),  Tˢ, Ξ  )
x10 = rand(Uniform(2.8,4.3),  Tˢ, Ξ  )
x11 = rand(Uniform(2.7,4.2),  Tˢ, Ξ  )
x12 = rand(Uniform(2.0,3.0),  Tˢ, Ξ  )

dξ = round.([reshape(x1, 1, Tˢ, :);
            reshape(x2, 1, Tˢ, :);
            reshape(x3, 1, Tˢ, :);
            reshape(x4, 1, Tˢ, :);
            reshape(x5, 1, Tˢ, :);
            reshape(x6, 1, Tˢ, :);
            reshape(x7, 1, Tˢ, :);
            reshape(x8, 1, Tˢ, :);
            reshape(x9, 1, Tˢ, :);
            reshape(x10, 1, Tˢ, :);
            reshape(x11, 1, Tˢ, :);
            reshape(x12, 1, Tˢ, :);
            ]
            ,digits=1)


##########################
# varaible stage 1
##########################
@variable(skill, ψ[1:H , 1:J], Bin)
@variable(skill, 0 ≤ χ[1:H , 1:J] ≤ 1)
@variable(skill, α[1:H , 1:J , 1:Tᵈ], Bin)
@variable(skill, z1[1:H , 1:J , 1:Tᵈ], Bin)
@variable(skill, z2[1:H , 1:J , 1:Tᵈ]);


##########################
# varaible stage 2
##########################
@variable(skill, 0 ≤ γξ[1:J , 1:Tˢ , 1:Ξ])
@variable(skill, αξ[1:H , 1:J , 1:Tˢ , 1:Ξ], Bin)
@variable(skill, z3[1:H , 1:J , 1:Tˢ , 1:Ξ], Bin)
@variable(skill, z4[1:H , 1:J , 1:Tˢ , 1:Ξ])



@constraint(skill, ψ[1,1] == 1)
@constraint(skill, ψ[2,1] == 1)
@constraint(skill, ψ[3,1] == 1)

@constraint(skill, ψ[4,2] == 1)
@constraint(skill, ψ[5,2] == 1)
@constraint(skill, ψ[6,2] == 1)

@constraint(skill, ψ[7,3] == 1)
@constraint(skill, ψ[8,3] == 1)
@constraint(skill, ψ[9,3] == 1)

@constraint(skill, ψ[10,4] == 1)
@constraint(skill, ψ[11,4] == 1)
@constraint(skill, ψ[12,4] == 1)

@constraint(skill, ψ[13,5] == 1)
@constraint(skill, ψ[14,5] == 1)
@constraint(skill, ψ[15,5] == 1)

@constraint(skill, ψ[16,6] == 1)
@constraint(skill, ψ[17,6] == 1)
@constraint(skill, ψ[18,6] == 1)

@constraint(skill, ψ[19,7] == 1)
@constraint(skill, ψ[20,7] == 1)
@constraint(skill, ψ[21,7] == 1)

@constraint(skill, ψ[22,8] == 1)
@constraint(skill, ψ[23,8] == 1)
@constraint(skill, ψ[24,8] == 1)

@constraint(skill, ψ[25,9] == 1)
@constraint(skill, ψ[26,9] == 1)
@constraint(skill, ψ[27,9] == 1)

@constraint(skill, ψ[28,10] == 1)
@constraint(skill, ψ[29,10] == 1)
@constraint(skill, ψ[30,10] == 1)

@constraint(skill, ψ[31,11] == 1)
@constraint(skill, ψ[32,11] == 1)
@constraint(skill, ψ[33,11] == 1)

@constraint(skill, ψ[34,12] == 1)
@constraint(skill, ψ[35,12] == 1)
@constraint(skill, ψ[36,12] == 1);



s = fill(0, 48, 12)
s[1,1] = 1
s[2,1] = 1
s[3,1] = 1

s[4,2] = 1
s[5,2] = 1
s[6,2] = 1

s[7,3] = 1
s[8,3] = 1
s[9,3] = 1


s[10,4] = 1
s[11,4] = 1
s[12,4] = 1

s[13,5] = 1
s[14,5] = 1
s[15,5] = 1

s[16,6] = 1
s[17,6] = 1
s[18,6] = 1

s[19,7] = 1
s[20,7] = 1
s[21,7] = 1

s[22,8] = 1
s[23,8] = 1
s[24,8] = 1

s[25,9] = 1
s[26,9] = 1
s[27,9] = 1


s[28,10] = 1
s[29,10] = 1
s[30,10] = 1

s[31,11] = 1
s[32,11] = 1
s[33,11] = 1

s[34,12] = 1
s[35,12] = 1
s[36,12] = 1

s[37,1] = 1
s[38,2] = 1
s[39,3] = 1
s[40,4] = 1
s[41,5] = 1
s[42,6] = 1
s[43,7] = 1
s[44,8] = 1
s[45,9] = 1
s[46,10] = 1
s[47,11] = 1
s[48,12] = 1

s;

zᵖ = [
    180.16 ; #Laboring
    180.16 ; #Laboring
    180.16 ; #Laboring
    180.16 ; #Caulkering
    180.16 ; #Caulkering
    180.16 ; #Caulkering
    219.28 ; #Mechanical controlling
    219.28 ; #Mechanical controlling
    219.28 ; #Mechanical controlling
    195.2 ; #Tiling
    195.2 ; #Tiling
    195.2 ; #Tiling
    224.96 ; #Plumbing
    224.96 ; #Plumbing
    224.96 ; #Plumbing
    219.28 ; #Plastering
    219.28 ; #Plastering
    219.28 ; #Plastering
    234.64 ; #Carpentry
    234.64 ; #Carpentry
    234.64 ; #Carpentry
    243.52 ; #Electric work
    243.52 ; #Electric work
    243.52 ; #Electric work
    220.16 ; #Water- proofing 
    220.16 ; #Water- proofing 
    220.16 ; #Water- proofing 
    189.92 ; #Glazing
    189.92 ; #Glazing
    189.92 ; #Glazing
    234.64 ; #Joining
    234.64 ; #Joining
    234.64 ; #Joining
    205.76 ; #Painting
    205.76 ; #Painting
    205.76 ; #Painting
    180.16 ; #Laboring
    180.16 ; #Caulkering
    219.28 ; #Mechanical controlling
    195.2 ; #Tiling
    224.96 ; #Plumbing
    219.28 ; #Plastering
    234.64 ; #Carpentry
    243.52 ; #Electric work
    220.16 ; #Water- proofing 
    189.92 ; #Glazing
    234.64 ; #Joining
    205.76 ; #Painting
] * 249;


zᶜ = [
    180.16 ; #1_Laboring
    180.16 ; #2_Caulkering
    219.28 ; #3_Mechanical controlling
    195.20 ; #4_Tiling
    224.96 ; #5_Plumbing
    219.28 ; #6_Plastering
    234.64 ; #7_Carpentry
    243.52 ; #8_Electric work
    220.16 ; #9_Water- proofing 
    189.92 ; #10_Glazing
    234.64 ; #11_Joining
    205.76 #12_Painting
] * 249 * 2;

zᵐ = fill(12000 , 48 , 12)
#1_ Tiler to be multi-skilled in plastering 
zᵐ[10 , 6] = 4000
zᵐ[11 , 6] = 4000
zᵐ[12 , 6] = 4000

#2_ Plumber to be multi-skilled in the caulking 
zᵐ[13 , 2] = 4000
zᵐ[14 , 2] = 4000
zᵐ[15 , 2] = 4000

#3_ Plasterer workforce to be multi-skilled in the tiling
zᵐ[16 , 4] = 4000
zᵐ[17 , 4] = 4000
zᵐ[18 , 4] = 4000

#4_ Carpenter to be multi-skilled in the joining
zᵐ[19 , 11] = 4000
zᵐ[20 , 11] = 4000
zᵐ[21 , 11] = 4000

#5_ Joiner to be multi-skilled in the Carpentering 
zᵐ[31 , 7] = 4000
zᵐ[32 , 7] = 4000
zᵐ[33 , 7] = 4000

#6_ Painter to be multi-skilled in the glazing
zᵐ[34 , 10] = 4000
zᵐ[35 , 10] = 4000
zᵐ[36 , 10] = 4000

zᵐ;


χᵐᵃˣ = fill(0.3, 48 ,12)

#1_ Tiler to be multi-skilled in plastering 
χᵐᵃˣ[10 , 6] = 0.7
χᵐᵃˣ[11 , 6] = 0.7
χᵐᵃˣ[12 , 6] = 0.7

#2_ Plumber to be multi-skilled in the caulking 
χᵐᵃˣ[13 , 2] = 0.7
χᵐᵃˣ[14 , 2] = 0.7
χᵐᵃˣ[15 , 2] = 0.7

#3_ Plasterer workforce to be multi-skilled in the tiling
χᵐᵃˣ[16 , 4] = 0.7
χᵐᵃˣ[17 , 4] = 0.7
χᵐᵃˣ[18 , 4] = 0.7

#4_ Carpenter to be multi-skilled in the joining
χᵐᵃˣ[19 , 11] = 0.7
χᵐᵃˣ[20 , 11] = 0.7
χᵐᵃˣ[21 , 11] = 0.7

#5_ Joiner to be multi-skilled in the Carpentering 
χᵐᵃˣ[31 , 7] = 0.7
χᵐᵃˣ[32 , 7] = 0.7
χᵐᵃˣ[33 , 7] = 0.7

#6_ Painter to be multi-skilled in the glazing
χᵐᵃˣ[34 , 10] = 0.7
χᵐᵃˣ[35 , 10] = 0.7
χᵐᵃˣ[36 , 10] = 0.7

#Plumbing j = 6 and electric work j = 8 

for h in 1:48
    χᵐᵃˣ[h,5] = 0
end

for h in 1:48
    χᵐᵃˣ[h,8] = 0
end

χᵐᵃˣ;


@objective(skill, Min, sum( zᵖ[h] * ψ[h,j] * (Tᵈ + Tˢ) for h in 1:H for j in 1:J ) +
                            sum(zᵐ[h,j] * χ[h,j] for h in 1:H for j in 1:J ) +
                            sum(zᶜ[j] * γξ[j,t,ξ] for j in 1:J for t in 1:Tˢ for ξ in 1:Ξ )
        )

@constraint(skill, linear1[h = 1:H , j = 1:J , t =1:Tᵈ], z1[h,j,t] ≤ α[h,j,t] )
@constraint(skill, linear2[h = 1:H , j = 1:J , t =1:Tᵈ], z1[h,j,t] ≤ ψ[h,j] )
@constraint(skill, linear3[h = 1:H , j = 1:J , t =1:Tᵈ], z1[h,j,t] ≥ α[h,j,t] + ψ[h,j] -1 )

@constraint(skill, linear4[h = 1:H , j = 1:J , t =1:Tᵈ], z2[h,j,t] ≤ α[h,j,t] )
@constraint(skill, linear5[h = 1:H , j = 1:J , t =1:Tᵈ], z2[h,j,t] ≤ χ[h,j] )
@constraint(skill, linear6[h = 1:H , j = 1:J , t =1:Tᵈ], z2[h,j,t] ≥ α[h,j,t] + χ[h,j] -1 )

@constraint(skill, demand_met[ j in 1:J , t in 1:Tᵈ ], sum(z1[h,j,t] + z2[h,j,t] for h in 1:H ) ≥ d[j,t] )
@constraint(skill, permanent_pool[h in 1:H , j in 1:J], ψ[h,j] ≤ s[h,j] )
@constraint(skill, allocation_recruitment_training[h in 1:H , j in 1:J , t in 1:Tᵈ], α[h,j,t] ≤ ψ[h,j] + 10 * χ[h,j] )
@constraint(skill, no_more_than_one_station[ h in 1:H , t in 1:Tᵈ ], sum(α[h,j,t] for j in 1:J) ≤ 1 )
@constraint(skill, single_skilling_at_recruitment[h in 1:H , j in 1:J], χ[h,j] ≤ ( 1 - ψ[h,j] ) * χᵐᵃˣ[h,j] )
@constraint(skill, training_recruited[ h in 1:H ], sum(χ[h,j] for j in 1:J ) ≤ sum(ψ[h,j] for j in 1:J) * J  )

@constraint(skill, linear7[h = 1:H , j = 1:J , t =1:Tˢ, ξ = 1:Ξ], z3[h,j,t,ξ] ≤ αξ[h,j,t,ξ] )
@constraint(skill, linear8[h = 1:H , j = 1:J , t =1:Tˢ, ξ = 1:Ξ], z3[h,j,t,ξ] ≤ ψ[h,j] )
@constraint(skill, linear9[h = 1:H , j = 1:J , t =1:Tˢ, ξ = 1:Ξ], z3[h,j,t,ξ] ≥ αξ[h,j,t,ξ] + ψ[h,j] -1 )

@constraint(skill, linear10[h = 1:H , j = 1:J , t =1:Tˢ, ξ = 1:Ξ], z4[h,j,t,ξ] ≤ αξ[h,j,t,ξ] )
@constraint(skill, linear11[h = 1:H , j = 1:J , t =1:Tˢ, ξ = 1:Ξ], z4[h,j,t,ξ] ≤ χ[h,j] )
@constraint(skill, linear12[h = 1:H , j = 1:J , t =1:Tˢ, ξ = 1:Ξ], z4[h,j,t,ξ] ≥ αξ[h,j,t,ξ] + χ[h,j] -1 )
        
@constraint(skill, stage2_demand_met[j = 1:J , t =1:Tˢ, ξ = 1:Ξ], sum(z3[h,j,t,ξ] for h in 1:H ) + sum(z4[h,j,t,ξ] for h in 1:H) + γξ[j,t,ξ] ≥ dξ[j,t,ξ] )
@constraint(skill, stage2_permanent_allocability[h in 1:H , j in 1:J , t in 1:Tˢ, ξ = 1:Ξ], αξ[h,j,t,ξ] ≤ ψ[h,j] + 10 * χ[h,j] )
@constraint(skill, stage2_no_more_than_one_station[ h in 1:H , t in 1:Tˢ , ξ = 1:Ξ], sum(αξ[h,j,t,ξ] for j in 1:J) ≤ 1 )

optimize!(skill)

println("objective value : ", objective_value(skill))

df_ψ = DataFrame(value.(ψ), :auto)
@show df_ψ 

df_χ = DataFrame(value.(χ), :auto)
@show df_χ

println("total number of recruited workforce : ", sum(value.(ψ)) )

println("total amount of secondary skills : ", sum(round.(value.(χ), digits = 2) ) )
