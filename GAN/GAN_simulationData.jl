using Random
using Statistics
using Statistics: mean, median
using PyPlot
using Flux
using Flux: throttle, params
using BSON: @load, @save
using Revise
using CSV
using Gadfly
using Zygote
using LinearAlgebra
using Distributions
using JLD2

includet("C:\\Users\\farhadyar\\Desktop\\Helper.jl")



p = 21
n = 2500
eta = 0.00001
batch_size = 50


x = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Simulation Design\\data_scenario1_Bimodal_Binary.csv", header=true))



# data_withheader = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Simulation Design\\data_scenario1_removedBinary_addedBimodal2.csv", header=false))

data_withheader = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Simulation Design\\data_scenario1_Bimodal_Binary.csv", header=false))


header = data_withheader[1, 1:21]


# notice that the i , j is kind of confusing here but dont change it try to understand that.




Random.seed!(11)

batch_size = 100

x_temp = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\ist.csv", header=true))

data_withheader = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\ist.csv", header=false))

p = size(x_temp)[2] +1

n = size(x_temp)[1]
header = data_withheader[1, 2:p-1]
header = vcat("RCONSC1", "RCONSC2", header)


cnt = 1

x = fill(0, n-count(x_temp[:,p-1].=="NA"), p)
for i = 1:n
    if x_temp[i,p-1]!="NA"
        if x_temp[i,1]==0
            x[cnt,1] = 0
            x[cnt,2] = 0
        elseif x_temp[i,1]==1
            x[cnt,1] = 1
            x[cnt,2] = 0
        else
            x[cnt,1] = 0
            x[cnt,2] = 1
        end
        #println("chie")
        x[cnt,3:p-1] = x_temp[i,2:p-2]
        x[cnt,p] = Base.parse(Int64, x_temp[i,p-1])
        global cnt+=1
    end
end

n = size(x)[1]




# notice that the i , j is kind of confusing here but dont change it try to understand that.
dataTypeArray = fill("Binary", p)
for i = 1:p
    for j = 1:n
        if x[j,i] isa String
            dataTypeArray[i] = "String"
            break
        elseif x[j,i]!=0 && x[j,i]!=1
            dataTypeArray[i] = "Continuous"
            break
        end
    end
end


dataTypeArray





# x = generate_data(n=n, p=p)
# f, (ax1, ax2) = plt.subplots(1,2,figsize=(10,3))
# ax1.spy(collect(transpose(x[1:100,:])))
# cax = ax2.imshow(oddsratio(x[:,1:20]))
# f.colorbar(cax, ax=ax2)
# f.tight_layout()
#
# # summary statistics over train data
# I = block_index(1,5,1,5)
# J = block_index(1,5,6,10)
# K = block_index(6,10,6,10)
# L = block_index(1,10,11,50)
# M = block_index(11,50,11,50)
# σ², µ = variance(() -> log.(oddsratio(generate_data(n=n, p=p))), num_samples = 100)
# f, (ax1, ax2, ax3, ax4) = plt.subplots(2,2,figsize=(10,5))
# cax1 = ax1.imshow(µ[1:20,1:20])
# f.colorbar(cax1, ax=ax1)
# ax3.boxplot([µ[M],µ[L],µ[K],µ[J],µ[I]], vert=false)
# cax2 = ax2.imshow(σ²[1:20,1:20])
# f.colorbar(cax2, ax=ax2)
# ax4.boxplot([σ²[M],σ²[L],σ²[K],σ²[J],σ²[I]], vert=false)
# f.tight_layout()

#   GAN adapted from
#   https://github.com/smidl/AnomalyDetection.jl/blob/master/src/gan.jl
#
# p = 2
# n = 10000
# eta = 0.00001
# batch_size = 50
#
# Random.seed!(11)
# x1 = initialize_skewed(p,n)
#
# Random.seed!(11)
# x2 = initialize_bimodal(p,n)
#
# x = hcat(x1[:,1], x2[:,2])
# #
#

dataTypeArray = fill("Binary", p)
for i = 1:p
    for j = 1:n
        if x[j,i] isa String
            dataTypeArray[i] = "String"
            break
        elseif x[j,i]!=0 && x[j,i]!=1
            dataTypeArray[i] = "Continuous"
            break
        end
    end
end


dataTypeArray




x_st = fill(0.0, n, p)

x_std= std(x, dims=1)

x_mean = mean(x, dims=1)


for i =1:p
    if dataTypeArray[i] != "Continuous"
        println("not continuous")
        x_st[:,i] = x[:,i]
    else
        x_st[:,i] =  (x[:,i] .- x_mean[i])./(2*x_std[i])
    end
end



Random.seed!(11)

loss = [Float64[],Float64[]]
gen_error = Float64[]


struct GAN
    g # generator
    gg # non-trainable generator copy
    d # discriminator
    dd # non-trainable discriminator copy
    pz # code distribution
end

Flux.@treelike GAN

freeze(m) = Flux.mapleaves(Flux.data,m)


GAN(G::Flux.Chain, D::Flux.Chain; pz=randn) = GAN(G, freeze(G), D, freeze(D), pz)

getcode(gan::GAN) = Float64.(gan.pz(size(((Flux.params(gan.g)).order)[1],2)))
getcode(gan::GAN, n::Int) = Float64.(gan.pz(size(((Flux.params(gan.g)).order)[1],2), n))
generate(gan::GAN) = activation_gen(gan.g(getcode(gan)))
generate(gan::GAN, n::Int) = activation_gen(gan.g(randn(15,n)))

Dloss(gan::GAN, X, Z) = (- Float64(0.5)*(Statistics.mean(log.(gan.d(X) .+ eps(Float64))) + Statistics.mean(log.(1 .- gan.d(activation_gen(gan.gg(Z))) .+ eps(Float64)))))
Gloss(gan::GAN, Z) = (- Statistics.mean(log.(gan.dd(activation_gen(gan.g(Z))) .+ eps(Float64))))

# generator = Chain(Dense(10, 20, NNlib.leakyrelu), Dense(20, p, NNlib.σ))

#for 2 dimensions
# generator = Chain(Dense(2, 10, NNlib.leakyrelu), Dense(10, p))
# discriminator = Chain(Dense(p, 10, NNlib.leakyrelu), Dense(10, 1, NNlib.σ))

#

#ist
# generator = Chain(Dense(15, 30, NNlib.leakyrelu), Dense(30, p))
# discriminator = Chain(Dense(p, 25, NNlib.leakyrelu), Dense(25, 1, NNlib.σ))



generator = Chain(Dense(15, 30, NNlib.leakyrelu), Dense(30, p))
discriminator = Chain(Dense(p, 25, NNlib.leakyrelu), Dense(25, 1, NNlib.σ))





gan = GAN(generator, discriminator, pz=randn)

# train gan



function eval_gen(gan::GAN)
    sum((Statistics.mean(x_st,dims=1) .- Statistics.mean(generate(gan,n) .- rand(p,n),dims=2)).^2)
end


train1(gan, 200, batch_size)

# f, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
# ax1.plot(loss[1])
# ax1.plot(loss[2])
# ax2.plot(gen_error)


Random.seed!(11)
x_gen = transpose(generate(gan,n))

for i = 1:p
    if dataTypeArray[i] == "Binary"
        x_gen[:,i] = 1.0 * x_gen[:,i].>0.5
    end
end


x_gen_des = fill(0.0,n,p)
for i =1:p
    if dataTypeArray[i] != "Continuous"
        println("not continuous")
        x_gen_des[:,i] = x_gen[:,i]
    else
        x_gen_des[:,i] =  (x_gen[:,i] .* (2*x_std[i])) .+ x_mean[i]
    end
end



set_default_plot_size(50cm, 50cm)
histogram_matrix(x, round_discrete(x_gen_des),"GAN")

# histogram_matrix(x_st, x_gen,"GAN")
savefig("C:\\Users\\farhadyar\\Desktop\\Histogram_GAN_SimulationDesign.png")

# set_default_plot_size(5cm, 5cm)
plot_temp = histogram_dimensions(x, round_discrete(x_gen_des),"GAN")
save_plot(plot_temp, "C:\\Users\\farhadyar\\Desktop\\Dimensions_GAN_SimulationDesign.svg")



set_default_plot_size(5cm, 5cm)

loss_plot(loss[1])

loss_plot(loss[2])

loss_plot(gen_error)



writedlm( "C:\\Users\\farhadyar\\Desktop\\Simulation Design data\\GAN\\Synthetic Data\\syndata.csv",  round_discrete(syndata_posterior), ',')





#
# JLD2.@save "Generator_2D.jld" Flux.params(gan.g)
# JLD2.@save "Discriminator_2D.jld" Flux.params(gan.d)
#
#
# gan2 = GAN(generator, discriminator, pz=randn)
#
# Flux.loadparams!(Flux.params(gan2.g), "Generator_2D.jld" )
# Flux.loadparams!(Flux.params(gan2.d), "Discriminator_2D.jld" )
#
# Random.seed!(11)
# x_gen2 = transpose(generate(gan2,10000))
# histogram_dimensions(x_gen, x_gen2,"")



# plot evaluation
x_gen = collect(transpose((1.0 .* (generate(gan,100) .> 0.5))))
OR = oddsratio(x_gen)
cl1, cl2 = k_means_sort(x_gen,2,100)
σ², µ = variance(() -> log.(oddsratio(collect(transpose((1.0 .* (generate(gan,100) .> 0.5)))))), num_samples = 100)
I = block_index(1,5,1,5);
J = block_index(1,5,6,10);
K = block_index(6,10,6,10);
L = block_index(1,10,11,50);
M = block_index(11,50,11,50);
f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(4,2,figsize=(10,15))
ax1.plot(loss[1])
ax1.plot(loss[2])
ax5.plot(gen_error)
ax2.spy(collect(transpose(vcat(cl1[1:min(end,50),:],cl2[1:min(end,50),:]))))
cax6 = ax6.imshow(min.(10,OR[1:20,1:20]))
ax6.text(1,1,string(median(OR[I])),color="white")
ax6.text(6,2.5,string(median(OR[J])),color="white")
ax6.text(6,6,string(median(OR[K])),color="white")
ax6.text(12,4,string(median(OR[L])),color="white")
ax6.text(12,12,string(median(OR[M])),color="white")
f.colorbar(cax6, ax=ax6)
cax3 = ax3.imshow(min.(10,µ[1:20,1:20]))
f.colorbar(cax3, ax=ax3)
cax4 = ax4.imshow(min.(10,σ²[1:20,1:20]))
f.colorbar(cax4, ax=ax4)
ax7.boxplot([µ[M],µ[L],µ[K],µ[J],µ[I]], vert=false, showfliers=false)
cax8 = ax8.boxplot([σ²[M],σ²[L],σ²[K],σ²[J],σ²[I]], vert=false, showfliers=false)

# save summary statistics
@save string("results/","02_04_","gan_n_",n,"_variance.bson") σ²
@save string("results/","02_04_","gan_n_",n,"_mean.bson") µ
@save string("results/","02_04_","gan_n_",n,"_x_gen.bson") x_gen
