g(X) = (h = A(X); (latentμ(h), latentlogσ(h)))

latentz(μ, logσ) = μ + exp.(logσ) * randn()

# Generative model / "decoder" MLP.

# KL-divergence between approximation posterior and N(0, 1) prior.
kl_q_p(μ, logσ) = (klp = 0.5 * sum(exp.(2 .* logσ) + μ.^2 .- 1 .- (2 .* logσ)); klp)

logp_x_z(x, z) = (fval = f(z); s =sum(log.(pdf.(Normal.(mu(fval),sig(fval)), (x'[:, dataTypeArray .== "Continuous"])') .+eps(Float32))); s)

logpdf1(b::Bernoulli, y) = ( y * log(b.p + eps(Float32)) + (1f0 - y) * log(1 - b.p + eps(Float32)))

logp_x_z_bernouli(x, z) = (s = sum(logpdf1.(Bernoulli.(bernoulli_pi(f(z))), (x'[:, dataTypeArray .== "Binary"])')); s)

# Monte Carlo estimator of mean ELBO using M samples.
function L̄(X) 
    (μ̂, logσ̂) = g(X)
    
    latz = latentz.(μ̂, logσ̂)
    
    if size(x_binary)[2] ==0
        lower_bound = (logp_x_z(X , latz) - kl_q_p(μ̂, logσ̂)) / m
    elseif size(x_continuous)[2] ==0
        lower_bound = (logp_x_z_bernouli(X , latz) - kl_q_p(μ̂, logσ̂)) / m
    else
        lower_bound = (logp_x_z(X , latz) + count(dataTypeArray .=="Binary")/count(dataTypeArray .=="Continuous")*logp_x_z_bernouli(X , latz) - kl_q_p(μ̂, logσ̂)) / m
    end
end

function loss(X)
    if size(x_binary)[2] ==0
        lo = -L̄(X) + 0.01f0 * sum(x->sum(x.^2),Flux.params(f, mu , sig))
    elseif size(x_continuous)[2] ==0
        lo = -L̄(X) + 0.01f0 * sum(x->sum(x.^2),Flux.params(f, bernoulli_pi))
    else
        lo = -L̄(X) + 0.01f0 * sum(x->sum(x.^2),Flux.params(f, mu , sig, bernoulli_pi))
    end
    return lo
end

function trainVAE!(input, stat)
    Flux.loadparams!(params1, "initialModel.jld" )

    global status = stat

    data1 = [[input[k,:]'] for k in Iterators.partition(1:n,m)]
    
    Flux.@epochs 30 (Flux.train!(loss, params1, data1, opt); average_loss(data1))

    return params1
end

function VAE_output(title)

    Random.seed!(11)
    
    if status == "with_transformations"
        data1 = [[x_tr_power_st[k,:]'] for k in Iterators.partition(1:n,m)]
    else
        data1 = [[x_st[k,:]'] for k in Iterators.partition(1:n,m)]

    end

    if title == "prior"
        # use Standard Normal and producing these Zs instead of getting them from Original Data 

        truesig = fill(0.0,Dz,Dz)
        truesig[diagind(truesig)] .= 1.0
        truemu = fill(0.0,Dz)

        zmat_normal = (collect(rand(Distributions.MvNormal(truemu,truesig), n)'))

        zmat = zmat_normal
        gval = map(val -> g(val[1]),data1)
        latvals = map(val -> latentz(val...),gval)
        latvals_array = fill(0.0,n,2)
        for j = 1:Int(ceil(n/m))
            for i = 1:m
                if (j == Int(ceil(n/m))) 
                    if ((j-1)*m+i) == n+1
                        break
                    end
        
                else
                    latvals_array[((j-1)*m+i), 1] = latvals[j][1,i]
                    latvals_array[((j-1)*m+i), 2] = latvals[j][2,i]
                end
            end
        end

        latvals= [transpose(zmat[k,:]) for k in Iterators.partition(1:n,m)]

    elseif title =="posterior"
        gval = map(val -> g(val[1]),data1)
        latvals = map(val -> latentz(val...),gval)
    end

    Random.seed!(11)
    recvals = map(latvals) do val
        fval = f(val)
        if size(x_binary)[2] ==0
            continuous =rand.(Normal.(mu(fval),sig(fval)))
        elseif size(x_continuous)[2] ==0
            continuous = rand.(Bernoulli.(bernoulli_pi(fval)))
        else
            continuous =vcat(rand.(Normal.(mu(fval),sig(fval))), rand.(Bernoulli.(bernoulli_pi(fval))))
        end
    end

    temp_recvals =fill(0.0, n, p)
    for j = 1:(floor(Int,n/m))
        cnt_binary = p - size(x_binary)[2] + 1
        cnt_continuous = 1
        for i = 1:p
            if dataTypeArray[i] == "Binary"
                temp_recvals[(j-1)*m+1: j*m,i] =recvals[j][cnt_binary,:]
                cnt_binary +=1
            elseif dataTypeArray[i] == "Continuous"
                temp_recvals[(j-1)*m+1: j*m,i] =recvals[j][cnt_continuous,:]
                cnt_continuous +=1
            end
        end
    end

    if status == "with_transformations"
        for i=1:p
            if dataTypeArray[i] =="Continuous"
                temp_recvals[:,i] = (temp_recvals[:,i] .* 2*x_tr_power_std[i]) .+ x_tr_power_mean[i]
            end
        end

        x_retr_power=fill(0.0, n, p)
        for j =1:n
            x_retr_power[j,:] = power_backtransform(temp_recvals[j,:])
        end

        x_retr_BC =fill(0.0, n, p)
        for i=1:p
            if dataTypeArray[i] =="Continuous"
                x_retr_BC[:,i] = (x_retr_power[:,i] .* 2*x_tr_BC_std[i]) .+ x_tr_BC_mean[i]
            else
                x_retr_BC[:,i] = x_retr_power[:,i]
            end
            
        end

        x_retr =fill(0.0, n, p)
        for j =1:n
            x_retr[j,:] = bc_backtransform(x_retr_BC[j,:] )
        end

        for i=1:p
            if dataTypeArray[i] =="Continuous"
                x_retr[:,i] = (x_retr[:,i] .* 2*x_std[i]) .+ x_mean[i]
            end
        end

        return x_retr
    else

        for i=1:p
            if dataTypeArray[i] =="Continuous"
                temp_recvals[:,i] = (temp_recvals[:,i] .* 2*x_std[i]) .+ x_mean[i]
            end
        end
        return temp_recvals
    end
end


function quantile_VAE(title)

    Random.seed!(11)

    if status == "with_transformations"
        data1 = [[x_tr_power_st[k,:]'] for k in Iterators.partition(1:n,m)]
    else
        data1 = [[x_st[k,:]'] for k in Iterators.partition(1:n,m)]
    end

    if title == "prior"
        Random.seed!(11) # use Standard Normal and producing these Zs instead of getting them from Original Data 

        truesig = fill(0.0,Dz,Dz)
        truesig[diagind(truesig)] .= 1.0
        truemu = fill(0.0,Dz)

        zmat_normal = (collect(rand(Distributions.MvNormal(truemu,truesig), n)'))
        zmat = zmat_normal
        
    elseif title =="posterior"
        gval = map(val -> g(val[1]),data1)
        latvals = map(val -> latentz(val...),gval)
        zmat = collect(hcat(map(val -> val,latvals)...)')
    end

    includet("C:\\Users\\farhadyar\\Desktop\\AIQN\\AIQN.jl")

    order_ = find_Order(zmat)

    println(order_)

    Random.seed!(11)
    synval_Array = qfind(zmat, order_)

    latvals_synvals= [transpose(synval_Array[k,:]) for k in Iterators.partition(1:n,m)]

    synvals = map(latvals_synvals) do val
    fval = f(val)
    continuous =vcat(rand.(Normal.(mu(fval),sig(fval))), rand.(Bernoulli.(bernoulli_pi(fval))))
    end

    temp_synvals =fill(0.0, n, p)
    for j = 1:(floor(Int,n/m))
        cnt_binary = p - size(x_binary)[2] + 1
        cnt_continuous = 1
        for i = 1:p
            if dataTypeArray[i] == "Binary"
                temp_synvals[(j-1)*m+1: j*m,i] =synvals[j][cnt_binary,:]
                cnt_binary +=1
            elseif dataTypeArray[i] == "Continuous"
                temp_synvals[(j-1)*m+1: j*m,i] =synvals[j][cnt_continuous,:]
                cnt_continuous +=1
            end
        end
    end

    for i=1:p
        if dataTypeArray[i] =="Continuous"
            temp_synvals[:,i] = (temp_synvals[:,i] .* 2*x_std[i]) .+ x_mean[i]
        end
    end
    return temp_synvals

end

function average_loss(data)
    append!(loss_array_VAE, mean([loss(data[i][1]) for i=1:ceil(Int, n/m)]))
end