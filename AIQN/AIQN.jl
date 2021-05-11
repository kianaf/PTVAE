function findMean(tau)
    min = 1
    index = 0
    for i = 1:length(tau)
        dif = abs(tau[i]-0.5)
        if dif < min
            min = dif
            index = i
        else
        end
    end
    return index
end

function find1573(tau)
    min = 1
    index = 0
    for i = 1:length(tau)
        dif = abs(tau[i]-0.1573)
        dif2 = abs(tau[i]- 0.8427)
        if (dif < min)||(dif2 < min)
            min = dif
            index = i
        end
    end
    return index
end

# ******************************************************************
function find_Score!(first, second, zmat)

    shape = size(zmat)
    z_dimention = shape[2]
    #This is for quantiles
    synval_Array = fill(0.0, n, z_dimention)

    zmat_Copy = hcat(zmat[:, first], zmat[:, second])

    zindex = repeat(1:n,10)[randperm(n*10)]

    synval_Median = 0
    synval_conditional = []
    synval_second_Unconditional = 0

    comb_z1 = 0
    comb_z2 = 0

    for i = 1:2  # It is for Zj|Zi and Zi|Zj
        Random.seed!(11)
        tau = rand(n).*0.9 .+ 0.05
        tau1 = rand(n).*0.9 .+ 0.05

        for j = 1:2 # It is for qfind for Zi and qfind for Zj

            qfind_Network = Chain(Dense(j,3,tanh),Dense(3,1))

            if j ==1
                zval = zmat_Copy[repeat(1:n,10)[randperm(n*10)], 1]
                data = [(zval[k],[rand()]) for k in Iterators.partition(1:length(zval),1)]
            else
                data = [(zmat_Copy[zindex[k],:],[rand()]) for k in Iterators.partition(1:length(zindex),1)]
            end

            opt = ADAM(0.01)
            params = Flux.params(qfind_Network)

            function loss(z,tau)
                kappa = 0.1
                if (length(z)>1)
                    u = (z[length(z)] .- qfind_Network([tau[1], z[1:(length(z)-1)][1]]))[1]
                else
                    u = (z .- qfind_Network(tau))[1]
                end
                tauval = tau[1]
                condval = abs((tauval - ((u[1] <= 0.0) ? 1.0 : 0.0)))[1]

                if abs(u[1]) <= kappa
                    return (condval/(2*kappa))*u^2 #+ 0.01 * sum(x->sum(x.^2), Flux.params(qfind2))
                else
                    return condval*(abs(u) - 0.5*kappa)  #+ 0.01 * sum(x->sum(x.^2), Flux.params(qfind2))
                end
            end

            Flux.@epochs 5 Flux.train!(loss, params, data, opt)


            #Get unconditional second dimention and conditional second on first in median
            if j == 1
                synval_Median = map(val -> qfind_Network([val])[1], 0.5)
                synval_second_Unconditional =  map(val -> qfind_Network([val])[1], tau)
            else
                synval_conditional = [qfind_Network([tau1[k], synval_Median])[1] for k=1:n]
                println(n)
                println("what is happening here?")
            end
        end

        #This Second unconditional means Zsecond with out conditions and conditional means Zsecond|Zfirst
        d_Second_Unconditional = quantile.(Normal(synval_second_Unconditional[findMean(tau)], abs(synval_second_Unconditional[findMean(tau)]-synval_second_Unconditional[find1573(tau)])), tau)
        d_conditional = quantile.(Normal(synval_conditional[findMean(tau1)], abs(synval_conditional[findMean(tau1)]-synval_conditional[find1573(tau1)])), tau1)


        display(scatterplot(d_Second_Unconditional, synval_second_Unconditional))
        display(scatterplot(d_conditional, synval_conditional))

        p_value_unconditional = ApproximateTwoSampleKSTest(d_Second_Unconditional, synval_second_Unconditional)
        p_value_conditional = ApproximateTwoSampleKSTest(d_conditional, synval_conditional)

        println(p_value_conditional)
        println(pvalue(p_value_conditional))
        println(p_value_unconditional)
        println(pvalue(p_value_unconditional))

        # https://daithiocrualaoich.github.io/kolmogorov_smirnov/
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35g.htm

        if i ==1
            println("comb_P_value1")
            comb_z1 = - 2 * sum(log(pvalue(p_value_conditional)) + log(pvalue(p_value_unconditional)))
            println(comb_z1)
        else
            println("comb_P_value2")
            comb_z2 = - 2 * sum(log(pvalue(p_value_conditional)) + log(pvalue(p_value_unconditional)))
            println(comb_z2)
        end
        zmat_Copy = hcat(zmat[:, second], zmat[:, first])
    end
    return comb_z1, comb_z2
end
# ********************************************************************
function find_Order(zmat)
    shape = size(zmat)
    z_dimention = shape[2]

    # K_S is a metrix in each row we have score, first element and second element
    number_Of_Scores = Int(((z_dimention * z_dimention) - z_dimention)/ 2)
    K_S = fill(0.0, number_Of_Scores , 3)

    # Number of scores which has been added
    cnt = 1
    for i = 1:z_dimention
        for j = i+1:z_dimention
            score1, score2 = find_Score!(i,j,zmat)
            if (score1 > score2)
                K_S[cnt, 1] = score1
                K_S[cnt, 2] = i
                K_S[cnt, 3] = j
            else
                K_S[cnt, 1] = score2
                K_S[cnt, 2] = j
                K_S[cnt, 3] = i
            end
            cnt +=1
        end
    end

    order = []
    K_S[sortperm(K_S[:, 1]), :]

    for i = 1:size(K_S)[1]
        println("count")
        println(K_S[(size(K_S)[1])-i+1, 2])
        println(K_S[(size(K_S)[1])-i+1, 3])

        if K_S[(size(K_S)[1])-i+1, 2] in order || K_S[(size(K_S)[1])-i+1, 3] in order
            continue
        else
            append!(order, Int(K_S[size(K_S)[1]-i+1, 2]))
            append!(order, Int(K_S[size(K_S)[1]-i+1, 3]))
        end
    end

    if (length(order)==z_dimention)
        println("order")
        println(order)
        return order
    else
        for i = 1:z_dimention
            if i in order
                continue
            else
                append!(order, i)
            end
        end
        println("order")
        println(order)
        return order
    end

end
# ********************************************************************
function qfind(zmat, order)
    shape = size(zmat)
    z_dimention = shape[2]
    zmat_Copy = fill(0.0, n, 1)

    #This is for quantiles
    synval_Array = fill(0.0, n, z_dimention)

    for i =1:z_dimention
        Random.seed!(11)
        qfind_Network = Chain(Dense(i, 3, tanh), Dense(3, 1))
        # zindex = repeat(1:n,10)[randperm(n*10)]
        function loss(z,tau)
            kappa = 0.1
            if (length(z)>1)
                input_temp = fill(0.0, length(z)-1)
                input = Zygote.Buffer(input_temp, length(z), 1)
                input[1] = tau[1]
                # append!(input, tau)
                for j = 2:length(z)
                    # append!(input, z[j])
                    input[j] = z[j]
                end
                u = (z[length(z)] .- qfind_Network(copy(input)))[1]
            else
                u = (z .- qfind_Network(tau))[1]
            end
            tauval = tau[1]
            condval = abs((tauval - ((u[1] <= 0.0) ? 1.0 : 0.0)))[1]

            if abs(u[1]) <= kappa
                return (condval/(2*kappa))*u^2 #+ 0.01 * sum(x->sum(x.^2), Flux.params(qfind2))
            else
                return condval*(abs(u) - 0.5*kappa)  #+ 0.01 * sum(x->sum(x.^2), Flux.params(qfind2))
            end
        end

        if i ==1
            zmat_Copy = zmat[:, order[i]]
            zval = zmat_Copy[repeat(1:n,10)[randperm(n*10)], 1]
            data = [(zval[j],[rand()]) for j in Iterators.partition(1:length(zval),1)]
        else
            zmat_Copy = hcat(zmat_Copy, zmat[:, order[i]])
            zindex = repeat(1:n,10)[randperm(n*10)]
            data = [(zmat_Copy[zindex[j],:],[rand()]) for j in Iterators.partition(1:length(zindex),1)]
        end

        opt = ADAM(0.01)
        params = Flux.params(qfind_Network)

        Flux.@epochs 50 Flux.train!(loss, params, data, opt)


        Random.seed!(11)
        if i ==1
            synval_Array[:, 1] = [qfind_Network([rand().* 0.9 .+ 0.05])[1] for j=1:n]
        else
            synval_Array[:, i] = [qfind_Network(vcat(rand().* 0.9 .+ 0.05, synval_Array[j, 1:i-1]))[1] for j=1:n]
        end
        println("again")

    end
    return synval_Array
end
# ********************************************************************************************************************
