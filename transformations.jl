using Distributed
using Gadfly
using LinearAlgebra
using Random
using Distributions
using Flux
using Statistics
using StatsBase: geomean
using PyPlot
using Compose
using VegaLite
using DelimitedFiles
using Zygote
using Cairo
using Colors
using JLD2
using DistributionsAD
using DecisionTree
#######################################Producing different distributions for testing############################################
function initialize_Normal(p, n)
    Random.seed!(11)
    truesig = fill(0.0,p,p)
    truesig[diagind(truesig)] .= 1.0
    truemu = fill(0.0,p)
    x = (collect(rand(Distributions.MvNormal(truemu ,truesig),Int(n))'))
    x = x[randperm(n),:]
    return x
end

function initialize_skewed(p, n)
    truesig = fill(0.0,p,p)
    truesig[diagind(truesig)] .= 1.0
    truemu = fill(0.0,p)
    x = (collect(rand(Distributions.Gumbel(0,0.15),n)'))
    x = vcat(x,(collect(rand(Distributions.Gumbel(0,0.15),n)')))
    x= x'
    x = x[randperm(n),:]
    return x
end

function initialize_bimodal(p, n)
    truesig = fill(0.0,p,p)
    truesig[diagind(truesig)] .= 1.0
    truemu = fill(0.0,p)
    x = (collect(rand(Distributions.MvNormal(truemu .+0 ,truesig),Int(7n/10))'))
    x = vcat(x,collect(rand(Distributions.MvNormal(truemu .+ 4 ,truesig  ),Int(3n/10))'))
    x = x[randperm(n),:]
    return x
end
#######################################Preprocessing############################################
function remove_badrecord!(x)
    for i=1:size(x)[1]
        if (i > size(x)[1])
            break
        elseif count(x[i,:].=="NA")>0
                x = x[1:end .!=i , 1:end]
                println(size(x)[1])
        end
    end
    return x
end

function stringtoNumber!(x)
    for i=1:size(x)[2]
        if (x[1,i] isa String)
            x[:,i] = Base.parse.(Int64, x[:,i])
        end
    end
    return x
end
#######################################BoxCox Transformation############################################
# change abs(min) to -min 
function set_alpha(x)
    alphaArray = fill(0.0, p)
    for j=1:p
        if dataTypeArray[j] != "Continuous"
            println("not continuous")
            continue
        end
        min = minimum(x[:,j])
        max = maximum(x[:,j])
        if min>0 #it was 0
            alpha = 0
        else
            alpha = -(min) + 0 + (0.01 * (max-min))
        end
        alphaArray[j] = alpha
    end
    return alphaArray
end


function set_lambda!(lambdaArray, x)
    params2= Flux.params(lambdaArray)

    opt = ADAM(0.01)

    for i = 1:p
        Random.seed!(11)
        if dataTypeArray[i] != "Continuous"
            println("not continuous")
            continue
        end
        Random.seed!(11)
        Flux.@progress for j= 1: 500
            con = false
            if j > 2
                if !con
                    @info "Epoch $j"
                    trainBoxCox!(params2, opt,  i, x)
                else
                    break
                end
            else
                @info "Epoch $j"
                trainBoxCox!(params2, opt,  i, x)
            end
        end
    end
    lambdaArray
end


function BC_transform_one_dimension(x, lambda, alpha, i)
    tr = fill(0.0, n)

    x_tr = Zygote.Buffer(tr, length(tr)[1], 1)

    x_tr =  @. lambda ≈ 0 ? log(x + alpha) : (((x + alpha) ^ lambda) -1.0) / lambda

    return copy(x_tr)
end

function backtransform(x)
    x_retr = fill(0.0, p)

    for i=1:p
        if dataTypeArray[i] !="Continuous"
            x_retr[i] = x[i]
        else
            if ((Rational(1/lambdaArray[i])).den % 2 !=0 || ((((x[i])) * lambdaArray[i])+ 1.0)>=0)
                x_retr[i] = lambdaArray[i] ≈ 0 ? exp(x[i]) - alphaArray[i]  : ((((x[i] ) * lambdaArray[i]) + 1.0) ^ (1/lambdaArray[i])) - alphaArray[i]
            else
                x_retr[i] = lambdaArray[i] ≈ 0 ? exp(x[i]) - alphaArray[i]  : 0 - alphaArray[i]
            end
        end
    end
    return x_retr
end

function trainBoxCox!(params2, opt, i, x)
    try
        gs = gradient(params2) do
            -log_likelihood_BoxCox(x[:,i], i)
        end
        lastParam = lambdaArray[i]
        Flux.Optimise.update!(opt, params2, gs)
        if abs((lastParam - lambdaArray[i]) / lastParam) < 0.01
            con = true
        end
    catch ex
    end

    lossValue_lambda = -log_likelihood_BoxCox(x[:,i],i)

    append!(loss_array_lambda[i], lossValue_lambda)
end

function round_discrete(input)
    output = fill(0.0, n,p)
    for i = 1:p
        if count(x[:,i].%1 .!=0) ==0
            output[:,i] = round.(input[:,i])
        else
            output[:,i] = input[:,i]
        end
    end
    return output
end
#######################################Evaluation############################################
function utility(x, syn)
    Random.seed!(11)
    comb = vcat(syn, x)
    labelTsyn = fill(1,  size(comb)[1]-size(x)[1],1)
    labelTx = fill(0, size(x)[1],1)
    labelT =  vcat(labelTsyn, labelTx)

    model = DecisionTreeClassifier(max_depth=15, min_samples_leaf = 20)  # maximum depth should be tuned using cv

    fit!(model, comb, labelT[:,1])
    println(model)
    
    P = [predict_proba(model, comb[i,:]) for i=1:n+n]

    pMSE = (1/(n + n)) * sum([((P[i][2].- (n/(n+n))).^2) for i=1:n+n])
    # Up = (1/(n + n)) * sum([(abs(P[i][2].- (n/(n+n)))) for i=1:n+n])
    
    # tree = print_tree(model)

    println(pMSE)
    
    numberOfPer = 100
    pMSE_per = fill(0.0, numberOfPer)
    
    #permute label 
    for i = 1:numberOfPer
        PerLabelT = labelT[randperm(length(labelT))]
        model = DecisionTreeClassifier(max_depth=15, min_samples_leaf = 20)  # maximum depth should be tuned using cv
        # fit!(model, comb.data, labelT[:,1])
        fit!(model, comb, PerLabelT[:,1])
        P = [predict_proba(model, comb[j,:]) for j=1:n+n]
        # P = [predict_proba(model, comb.data[i,:]) for i=1:n+n]
        # predictedLabels = [predict(model, comb.data[i,:])> 0.50 ? 1.0 : 0.0 for i=1:n+n]
        pMSE_per[i] = (1/(n + n)) * sum([((P[j][2].- (n/(n+n))).^2) for j=1:n+n])
    end
    
    pMSE_per_mean = mean(pMSE_per)
    pMSE_per_std = std(pMSE_per)
    
    pMSE_ratio = pMSE / pMSE_per_mean
    Standardize_pMSE = (pMSE - pMSE_per_mean)/pMSE_per_std

    println("pMSE_ratio: " ,pMSE_ratio)
    println("Standardize_pMSE: ", Standardize_pMSE)
end
#######################################Power function############################################
function set_x3ParametersArray!(shiftArray, peak1Array, peak2Array,powerArray, x)
    for i=1:p
        shiftArray[i] = medianArray[i]
        peak1Array[i] = -1
        peak2Array[i] = 1
    end
    
    for i = 1:p
        if dataTypeArray[i] != "Continuous"
            println("not continuous")
            continue
        end
        Random.seed!(11)
        Flux.@progress for j= 1: 5
            con = false
            if j > 2
                if !con
                    @info "Epoch $j"
                    trainx3!(shiftArray, peak1Array, peak2Array, powerArray, i, x,j)
                else
                    break
                end
            else
                @info "Epoch $j"
                trainx3!(shiftArray, peak1Array, peak2Array,powerArray, i, x,j)
            end
        end
        
        if IQR(x[:,i]) <= IQR(power_tr(x[:,i], i))
            powerArray[i] = 0
        end        
    end
    shiftArray, peak1Array, peak2Array, powerArray
end


function power_tr(x, i)
    
    x_tr = Zygote.Buffer(x, length(x)[1], 1)
    shift = shiftArray[i]   
    x_tr = x .- shift

    x_tr = (1 .- relu'.(x_tr .+ abs.(peak1Array[i]-shift))).* (-1 .- (abs.((x_tr .-(peak1Array[i] - shift))))) +
    (1 .- relu'.(x_tr)).*(relu'.(x_tr .+ abs.(peak1Array[i].-shift))) .*  ( x_tr ./ abs.(shift .- peak1Array[i])) +
    (relu'.(x_tr)) .* (1 .- relu'.(x_tr .- abs.(peak2Array[i].-shift)))  .*  ( x_tr ./ abs.(shift .- peak2Array[i])) +
    relu'.(x_tr .- abs.(peak2Array[i].-shift)) .* (1 .+  (abs.(peak2Array[i].-shift .- x_tr)))

    x_tr = (relu'.(x_tr) .- (1 .- relu'.(x_tr))).*((abs.(x_tr)).^(powerArray[i])) 

end

function power_tr_shift(x, i)
    x_tr = Zygote.Buffer(x, length(x)[1], 1)

    shift = shiftArray[i]   
    
    x_tr = x .- shift

    x_tr = (relu'.(x_tr) .- (1 .- relu'.(x_tr))).*((abs.(x_tr)).^(powerArray[i])) 
  
    return x_tr
end


function root_tr(x)
    x_retr = fill(0.0, p)
    for i=1:p
        if dataTypeArray[i] !="Continuous"
            x_retr[i] = x[i]
        elseif powerArray[i]==0
            x_retr[i] = x[i]
        else
            x_retr[i] = root(x[i], powerArray[i])
            shift = shiftArray[i]
            
            if x_retr[i]<= -1
                x_retr[i] = -abs(peak1Array[i] - shift - (((x_retr[i] + 1)/ (-1))))
            elseif -1 < x_retr[i] <= 0
                x_retr[i] =  (x_retr[i] / 1) * abs(shift - peak1Array[i])
            elseif 0 < x_retr[i] <= 1
                x_retr[i] =  (x_retr[i] / 1) * abs(shift - peak2Array[i]) 

            else
                x_retr[i] =  abs(((x_retr[i] - 1)/ (1)) + (peak2Array[i] - shift))
            end
            
            x_retr[i] = x_retr[i] + shift
        end
    end
    return x_retr
end


function root(x,r)
    (relu'(x)*x)^(1/r) - (1 - relu'(x))*(abs(x))^(1/r)
end


function trainx3!(shiftArray, peak1Array, peak2Array, powerArray, i, x, epoch)

    if epoch ==1
        shiftTrain = 50
    else
        shiftTrain = 10
    end

    for j=1:shiftTrain

        if epoch == 1
            params1 = Flux.params(shiftArray, powerArray)
        else
            params1 = Flux.params(shiftArray)
        end
        
        println(params1)
        try
            gs = gradient(params1) do
                if epoch == 1
                    (IQR(power_tr_shift(x[:,i], i)))                  
                else
                    (IQR(power_tr(x[:,i], i))) 
                end
            end

            opt = ADAM(0.01)

            lastParam = shiftArray[i]

            Flux.Optimise.update!(opt, params1, gs)         
            
            if abs((shiftArray[i]- lastParam)/ lastParam) < 0.01
                con == true
            end
        catch ex

        end
    end



    if epoch ==1
        powerTrain = 10
        peakTrain = 10
        
    else
        powerTrain = 10
        peakTrain = 10
    end
    
    
    for j=1:powerTrain
        params1 = Flux.params(powerArray)
        
        println(params1)
        try
            gs = gradient(params1) do
                if epoch == 1
                    (IQR(power_tr_shift(x[:,i], i)))
                else
                    (IQR(power_tr(x[:,i], i)))
                end
            end

            opt = ADAM(0.01)

            lastParam = powerArray[i]

            Flux.Optimise.update!(opt, params1, gs)         
            
            if abs((powerArray[i]- lastParam)/ lastParam) < 0.01
                con == true
            end
        catch ex

        end
    end

    for j = 1:peakTrain
        params1 = Flux.params(peak1Array)
        println(params1)
        try
            gs = gradient(params1) do
                (IQR(power_tr(x[:,i], i))) + 1000(floor(abs(peak1Array[i]/minArray[i])))*abs(peak1Array[i])
            end

            opt = ADAM(0.01)

            Flux.Optimise.update!(opt, params1, gs)
            
            if peak1Array[i]<minArray[i]
                peak1Array[i] = minArray[i]
            end            
        catch ex
        end
        lossValue = IQR(power_tr(x[:,i], i))  + 1000(floor(abs(peak1Array[i]/minArray[i])))*abs(peak1Array[i]) + 1000*(relu'(peak1Array[i] -shiftArray[i])*(abs(peak1Array[i] -shiftArray[i])))#+ abs(kurtosis(tanhInv_transform_one_dimension(x[:,i],tanhInvShiftArray, tanhInvScaleArray, i)))+ 1000*(relu'(peak1Array[i] -tanhInvShiftArray[i]) + relu'(tanhInvShiftArray[i] - peak2Array[i]))
        if (lossValue < minLoss[i])
            minLoss[i] = lossValue
        end
        append!(loss_array[i], lossValue)
    end
    for j = 1:peakTrain
        params1 = Flux.params(peak2Array)
        println(params1)
        try
            gs = gradient(params1) do
                (IQR(power_tr(x[:,i], i))) + 1000(floor(abs(peak2Array[i]/maxArray[i])))*abs(peak2Array[i])
            end
            opt = ADAM(0.01)

            Flux.Optimise.update!(opt, params1, gs)

            if peak2Array[i]>maxArray[i]
                peak2Array[i] = maxArray[i]
            end

        catch ex
    
        end
        lossValue = IQR(power_tr(x[:,i], i)) + 1000(floor(abs(peak2Array[i]/maxArray[i])))*abs(peak2Array[i])  + 1000* relu'(shiftArray[i] - peak2Array[i])*(abs(peak2Array[i] -shiftArray[i]))#+ abs(kurtosis(tanhInv_transform_one_dimension(x[:,i],tanhInvShiftArray, tanhInvScaleArray, i)))+ 1000*(relu'(peak1Array[i] -tanhInvShiftArray[i]) + relu'(tanhInvShiftArray[i] - peak2Array[i]))

        if (lossValue < minLoss[i])
            minLoss[i] = lossValue
        end
        append!(loss_array[i], lossValue)
    end
end
#######################################Criteria for Normality############################################
function IQR(x)
    iqr = (quantile(sort(x), 0.75,sorted = true) - (quantile(sort(x), 0.25,sorted = true)))
    Q3 = quantile(sort(x), 0.75,sorted = true)
    Q1 = quantile(sort(x), 0.25,sorted = true)
    Med = quantile(sort(x), 0.5,sorted = true)

    #68-95 rule symmetric ^^^^^^
    abs(quantile(sort(x), 0.8419,sorted = true) - Med - std(x)) + abs(Med - quantile(sort(x), 0.1581,sorted = true) - std(x)) + abs(quantile(sort(x), 0.1581,sorted = true) - quantile(sort(x), 0.0223,sorted = true) - std(x)) + abs(quantile(sort(x), 0.9777,sorted = true)-quantile(sort(x), 0.8419,sorted = true) - std(x))  
end

function bimodality_coefficient(x)
    return (skewness(x)^2 +1)/(kurtosis(x)+3) 
end

function log_likelihood(x)
    N = length(x)
    (-N / 2.0)*log.(Base.MathConstants.e,2π) + (-N / 2.0) *  log.(Base.MathConstants.e,(std(x))^2) - (1/2(std(x)^2))*sum((x .-mean(x)).^2)
end


function log_likelihood_BoxCox(x, i)
    N = length(x)
    y = BC_transform_one_dimension(x, lambdaArray[i], alphaArray[i], i)
    σ² = var(y, corrected = false) 
    -N / 2.0 * log(σ²) + (lambdaArray[i] - 1) * sum(log.(x .+ alphaArray[i])) 
end