cd("C:\\Users\\farhadyar\\Documents\\Project_PTVAE\\progs\\github_project\\PTVAE")

using Pkg;

Pkg.activate("v_env") #virtual environment activation for compatible package management

Pkg.status()


"""
Package versions should be:
  [336ed68f] CSV v0.5.24
  [159f3aea] Cairo v1.0.2
  [5ae59095] Colors v0.9.6
  [a81c6b42] Compose v0.8.1
  [a93c6f00] DataFrames v0.20.2
  [7806a523] DecisionTree v0.10.10
  [31c24e10] Distributions v0.21.12
  [ced4e74d] DistributionsAD v0.1.0
  [587475ba] Flux v0.10.4
  [c91e804a] Gadfly v1.2.1
  [09f84164] HypothesisTests v0.8.0
  [033835bb] JLD2 v0.1.11
  [e5e0dc1b] Juno v0.7.2
  [91a5bcdd] Plots v0.29.9
  [d330b81b] PyPlot v2.8.2
  [295af30f] Revise v2.5.2
  [60ddc479] StatPlots v0.9.2
  [2913bbd2] StatsBase v0.32.1
  [112f6efa] VegaLite v2.3.0
  [e88e6eb3] Zygote v0.4.20
  [8bb1440f] DelimitedFiles
  [8ba89e20] Distributed
  [37e2e46d] LinearAlgebra
  [9a3f8284] Random
  [10745b16] Statistics
  """

using Revise
using CSV

includet("transformations.jl")
includet("VAE.jl")
includet("visualization\\plotting_paper.jl")
includet("AIQN\\AIQN.jl")
includet("load_data.jl")


# ourMethod,FedSyn, VAE, QVAE, GAN = load_all_methods_sim!()
# ourMethod,FedSyn, VAE, GAN = load_all_methods_ist!()

# which dataset you want to load? 
# toy example data
# data_string  = "toy"

# simulation data
# data_string  = "sim"

# ist data
data_string  = "ist"

# your own data put the url
# data_string  = "url"

m, n, p, x, dataTypeArray = load_dataset(data_string)



############################ standardization #####################################
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


####################### Box-Cox transformation ##################################
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

############################ standardization of Box-Cox transformation output #####################################
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

####################### Power transformation ##################################
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

shiftArray, peak1Array, peak2Array, powerArray = set_power_parameter!(shiftArray, peak1Array, peak2Array, powerArray, x_tr_BC_st)

x_tr_power= fill(0.0, n,p)

for i =1:p
    if dataTypeArray[i] != "Continuous"
        println("not continuous")
        x_tr_power[:,i] = x[:,i]
    elseif powerArray[i] ==0
        x_tr_power[:,i] = x_tr_BC_st[:,i]
    else
        x_tr_power[:,i] = power_tr(x_tr_BC_st[:,i],i)
    end
end

scatterplot_matrix(x_tr_power, "")

x_retr_power =fill(0.0, n, p)

for j =1:n
    x_retr_power[j,:] = power_backtransform(x_tr_power[j,:]) 
end
scatterplot_matrix(x_retr_power, "")

for i= 1:p
    if dataTypeArray[i] != "Continuous"
        println("not continuous")
    else
        loss_plot(loss_array[i])
    end
end
############################ standardization of power transformation output #####################################
x_tr_power_st = fill(0.0, n,p)
x_tr_power_std= std(x_tr_power, dims=1)
x_tr_power_mean = mean(x_tr_power, dims=1)
quantileArray = fill(0.0,p)

for i =1:p
    if dataTypeArray[i] != "Continuous"
        println("not continuous")
        x_tr_power_st[:,i] = x_tr_power[:,i]
    else
        x_tr_power_st[:,i] =  (x_tr_power[:,i] .- x_tr_power_mean[i])./(2*x_tr_power_std[i])
    end
end

scatterplot_matrix(x_tr_power_st, "")


############################ VAE layers definition #####################################

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

params1 = trainVAE!(x_tr_power_st, "with_transformations")

loss_plot(loss_array_VAE)

##################Prior vs Posterior

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