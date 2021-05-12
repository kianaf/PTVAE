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
    
    println(pMSE)
    
    numberOfPer = 100
    pMSE_per = fill(0.0, numberOfPer)
    
    #permute labels 
    for i = 1:numberOfPer
        PerLabelT = labelT[randperm(length(labelT))]
        model = DecisionTreeClassifier(max_depth=15, min_samples_leaf = 20)  # maximum depth should be tuned using cv
        # fit!(model, comb.data, labelT[:,1])
        fit!(model, comb, PerLabelT[:,1])
        P = [predict_proba(model, comb[j,:]) for j=1:n+n]
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