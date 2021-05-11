# activate C:\Users\farhadyar\AppData\Local\Julia-1.3.1\Example\Project.toml
using Revise
using CSV
using DataFrames

includet("C:\\Users\\farhadyar\\Desktop\\transformations.jl")
includet("C:\\Users\\farhadyar\\Desktop\\VAE.jl")
includet("C:\\Users\\farhadyar\\Desktop\\plotting_paper.jl")
includet("C:\\Users\\farhadyar\\Desktop\\AIQN.jl")
includet("C:\\Users\\farhadyar\\Desktop\\load_parameters.jl")

Random.seed!(42)
m = 20
n = 5000
p = 2
x1 = initialize_skewed(p,n)
x2 = initialize_bimodal(p,n)
x = hcat(x1[:,1], x2[:,1])

ourMethod,FedSyn, VAE, QVAE, GAN = load_all_methods_sim!()
ourMethod,FedSyn, VAE, GAN = load_all_methods_ist!()

Random.seed!(11)
n=2500
p=21
m = 50

x = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Simulation Design\\data_scenario1_Bimodal_Binary.csv", header=true))

data_withheader = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Simulation Design\\data_scenario1_Bimodal_Binary.csv", header=false))

header = data_withheader[1, 1:p]

#### preprocess of IST removing NA and redifinition of RCONSC
Random.seed!(11)
m = 200
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

scatterplot_matrix(x, "Original Data")

Random.seed!(11)
x_st = fill(0.0, n,p)
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

scatterplot_matrix(x_st, "")

Random.seed!(11)

alphaArray = set_alpha(x_st)

loss_array_lambda = [push!([]) for i in 1:p]

LossValue_lambda = 0

lambdaArray = fill(0.5, p)

lambdaArray = set_lambda!(lambdaArray, x_st)

quantileArray_tr = fill(0.0, p)

x_tr_BC = fill(0.0, n,p)

for i =1:p
    if dataTypeArray[i] != "Continuous"
        println("not continuous")
        x_tr_BC[:,i] = x[:,i]
    else
        x_tr_BC[:,i] = BC_transform_one_dimension(x_st[:,i], lambdaArray[i], alphaArray[i], i)
    end
end

scatterplot_matrix(x_tr_BC, "BoxCox transformed")

for i= 1:p
    if dataTypeArray[i] != "Continuous"
        println("not continuous")
    else
        loss_plot(loss_array_lambda[i])
    end
end
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
x_tr_BC_st = fill(0.0, n,p)
x_tr_BC_std= std(x_tr_BC, dims=1)
x_tr_BC_mean = mean(x_tr_BC, dims=1)

for i =1:p
    if dataTypeArray[i] != "Continuous"
        println("not continuous")
        x_tr_BC_st[:,i] = x_tr_BC[:,i]
    else
        x_tr_BC_st[:,i] =  (x_tr_BC[:,i] .- x_tr_BC_mean[i])./(2*x_tr_BC_std[i])
    end
end

scatterplot_matrix(x_tr_BC_st, "")
########################################################################
Random.seed!(11)
maxArray = fill(0.0, p)
minArray = fill(0.0, p)
medianArray = fill(0.0, p)
meanArray = fill(0.0, p)
quantile1Array = fill(0.0, p)
quantile3Array = fill(0.0, p)

set_minmaxmedian(x_tr_BC_st)

minLoss =fill(10.0^10, p)
lossValue = 0.0

loss_array = [push!([]) for i in 1:p]

shiftArray, peak1Array, peak2Array, powerArray = set_x3ParametersArray!(shiftArray, peak1Array, peak2Array, powerArray, x_tr_BC_st)

x_tr_logit= fill(0.0, n,p)

for i =1:p
    if dataTypeArray[i] != "Continuous"
        println("not continuous")
        x_tr_logit[:,i] = x[:,i]
    elseif powerArray[i] ==0
        x_tr_logit[:,i] = x_tr_BC_st[:,i]
    else
        x_tr_logit[:,i] = power_tr(x_tr_BC_st[:,i],i)
    end
end

scatterplot_matrix(x_tr_logit, "")

x_retr_logit =fill(0.0, n, p)

for j =1:n
    x_retr_logit[j,:] = root_tr(x_tr_logit[j,:]) #.+ quantileArray_tr_logit)
end
scatterplot_matrix(x_retr_logit, "")

for i= 1:p
    if dataTypeArray[i] != "Continuous"
        println("not continuous")
    else
        loss_plot(loss_array[i])
    end
end
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
x_tr_logit_st = fill(0.0, n,p)
x_tr_logit_std= std(x_tr_logit, dims=1)
x_tr_logit_mean = mean(x_tr_logit, dims=1)
quantileArray = fill(0.0,p)

for i =1:p
    if dataTypeArray[i] != "Continuous"
        println("not continuous")
        x_tr_logit_st[:,i] = x_tr_logit[:,i]
    else
        x_tr_logit_st[:,i] =  (x_tr_logit[:,i] .- x_tr_logit_mean[i])./(2*x_tr_logit_std[i])
    end
end

scatterplot_matrix(x_tr_logit_st, "")

Random.seed!(11)

lossValue_VAE= 0.0

loss_array_VAE = []

x_binary =  x[:, dataTypeArray.=="Binary"]

x_continuous = x[:, dataTypeArray .== "Continuous"]

Dz, Dh, Dh2 = 3, p, p

Random.seed!(11)

A, latentμ, latentlogσ = Dense(p, Dh, tanh), Dense(Dh, Dz), Dense(Dh, Dz)


f = Dense(Dz, Dh, tanh)

mu = Dense(Dh,size(x_continuous)[2])

sig = Chain(Dense(Dh,size(x_continuous)[2]),arg -> exp.(arg))

bernoulli_pi = Dense(Dh, size(x_binary)[2], σ)

if size(x_binary)[2] ==0
    params1 = Flux.params(A, latentμ, latentlogσ, f, mu, sig)
elseif size(x_continuous)[2] ==0
    params1 = Flux.params(A, latentμ, latentlogσ, f, bernoulli_pi)
else
    params1 = Flux.params(A, latentμ, latentlogσ, f, mu, sig, bernoulli_pi)
end

JLD2.@save "initialModel.jld" params1

status = "not trained"

opt = ADAM(0.01)

params1 = trainVAE!(x_tr_logit_st, "with_transformations")

loss_plot(loss_array_VAE)

# it depends on which one do you want prior or posterior
#zmat = ???

set_default_plot_size(50cm, 50cm)

syndata_prior = VAE_output("prior")

scatterplot_matrix(round_discrete(syndata_prior), "")

histogram_all_dimensions(x, round_discrete(syndata_prior), "IST")

histogram_matrix(x, round_discrete(syndata_prior), "")

plot_temp = histogram_dimensions(x, round_discrete(syndata_prior), "Prior")

save_plot(plot_temp, "C:\\Users\\farhadyar\\Desktop\\Dimensions_ourVAE_prior.svg")

histogram_matrix(x, round_discrete(syndata_prior), string("Histogram_ourVAE_prior" , " " , status))
histogram_matrix(x, round_discrete(FedSyn), string("Histogram_ourVAE_prior" , " " , status))

savefig("C:\\Users\\farhadyar\\Desktop\\Histogram_ourVAE_prior.png")

syndata_posterior = VAE_output("posterior")


plot_temp = histogram_dimensions(x, round_discrete(syndata_posterior),  "posterior")


save_plot(plot_temp, "C:\\Users\\farhadyar\\Desktop\\Dimensions_histograms_ourVAE_posterior.svg")

histogram_matrix(x, round_discrete(syndata_posterior),"Histogram_ourVAE_posterior")

savefig("C:\\Users\\farhadyar\\Desktop\\Histogram_ourVAE_posterior.png")


# it depends on you which one do you want prior or posterior

#zmat = ???

syndata_prior_quantile = quantile_VAE("prior")

plot_temp = histogram_dimensions(x, syndata_prior_quantile, "syndata_prior_quantile")

save_plot(plot_temp, "C:\\Users\\farhadyar\\Desktop\\Dimensions_quantile&ourVAE_prior.svg")

histogram_matrix(x, syndata_prior_quantile, "Histogram_quantile&ourVAE_prior")

savefig("C:\\Users\\farhadyar\\Desktop\\Histogram_quantile&ourVAE_prior.png")

syndata_posterior_quantile = quantile_VAE("posterior")

plot_temp = histogram_dimensions(x, syndata_posterior_quantile, "syndata_posterior")

savefig("C:\\Users\\farhadyar\\Desktop\\Histogram_quantile&ourVAE_posterior.png")

#********************************************************************************************************
x_st = fill(0.0, n,p)
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

params1 = trainVAE!(x_st, "without_transformations")

syndata_prior = VAE_output("prior")

set_default_plot_size(50cm, 50cm)

plot_temp = histogram_dimensions(x, syndata_prior, string("syndata_prior" , " " , status))

save_plot(plot_temp, "C:\\Users\\farhadyar\\Desktop\\Dimensions_plainVAE_prior.svg")

histogram_matrix(x, syndata_prior, "Histogram_plainVAE_prior")

savefig("C:\\Users\\farhadyar\\Desktop\\Histogram_plainVAE_prior.png")

syndata_posterior = VAE_output("posterior")

plot_temp = histogram_dimensions(x, syndata_posterior, string("syndata_posterior", " ", status))

save_plot(plot_temp, "C:\\Users\\farhadyar\\Desktop\\Dimensions_histograms_plainVAE_posterior.svg")

histogram_matrix(x, syndata_posterior, "Histogram_plainVAE_posterior")

savefig("C:\\Users\\farhadyar\\Desktop\\Histogram_plainVAE_posterior.png")


############################################## AIQN ####################################################################
syndata_prior_quantile = quantile_VAE("prior")

plot_temp = histogram_dimensions(x, syndata_prior_quantile, string("syndata_prior_quantile", " ", status))

save_plot(plot_temp, "C:\\Users\\farhadyar\\Desktop\\Dimensions_quantileVAE_prior.svg")

histogram_matrix(x, syndata_prior_quantile, "Histogram_quantileVAE_prior")

savefig("C:\\Users\\farhadyar\\Desktop\\Histogram_quantileVAE_prior.png")

syndata_posterior_quantile = quantile_VAE("posterior")

plot_temp = histogram_dimensions(x, syndata_posterior_quantile, string("syndata_posterior_quantile", " ", status))

save_plot(plot_temp, "C:\\Users\\farhadyar\\Desktop\\Dimensions_quantileVAE_posterior.svg")

histogram_matrix(x, syndata_posterior_quantile, "Histogram_quantileVAE_posterior")

savefig("C:\\Users\\farhadyar\\Desktop\\Histogram_quantileVAE_posterior.png")