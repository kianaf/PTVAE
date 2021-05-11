
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
    x = (collect(rand(Distributions.MvNormal(truemu .+0 ,truesig),Int(5n/10))'))
    x = vcat(x,collect(rand(Distributions.MvNormal(truemu .+ 4 ,truesig  ),Int(5n/10))'))
    x = x[randperm(n),:]
    return x
end

function generate_data(;n=100,p=50)
    #x = 1.0*(rand(n,p) .> 0.9)
    #x[1:Int(n/2),1:5] .= 1
    #x[(Int(n/2)+1):end,6:10] .= 1
    x = 1.0*(rand(n,p) .> 0.9)
    z = fill(0.0,n,10)
    z[1:Int(n/2),1:5] .= 1
    z[(Int(n/2)+1):end,6:10] .= 1
    z = 1.0*(rand(n,10) .<= z*0.3)
    x[:,1:10] = 1.0*((x[:,1:10] .+ z) .> 0)
    x
end



function random_batch_index(x::AbstractArray, batch_size=1; dims=1)
    n = size(x,dims)
    Iterators.partition(shuffle(1:n), batch_size)
end


function oddsratio(x)
        N, dim = size(x)
        s = sum(x, dims=1)
        OR = zeros(dim,dim)
        for (i,j) in Iterators.product(1:dim,1:dim)
            if i != j
                a = N - sum(min.(x[:,i] .+ x[:,j], 1))      # ij_00
                diff = x[:,i] .- x[:,j]
                b = sum(1 .*( diff .== -1 ))                # ij_01
                c = sum(1 .*( diff .== 1 ))                 # ij_10
                d = sum(x[:,i] .* x[:,j])                   # ij_11
                OR[i,j] = (max(a,0.5)*max(d,0.5)/max(b,0.5)/max(c,0.5))
            end
        end
        return OR
end

function variance(draw::Function; num_samples = 10)
    x1 = draw()
    data = zeros(size(x1,1), size(x1,2), num_samples)
    data[:,:,1] = x1
    for i in 2:num_samples
        data[:,:,i] = draw()
    end
    µ = (sum(data, dims=3)./num_samples)[:,:,1]
    σ² = zeros(size(x1))
    for i in 1:num_samples
        σ² += (data[:,:,i] .- µ).^2
    end
    σ² = σ²./(num_samples - 1)
    return σ², µ
end

function block_index(i,j,k,l)
    I = Int[]
    for kdx in (k+1):l
        for idx in i:j
            if idx < kdx
                push!(I,idx + (kdx - 1)*p)
            end
        end
    end
    return I
end

function k_means_sort(x,K,epochs=100)
    norm_l1(x) = sum(abs.(x))

    function distance_matrix(X,means)
            # calculate Distance Matrix
        D = zeros(N,K)
        for (x,y) in Iterators.product(1:N,1:K)
            D[x,y] = norm_l1(means[y,:]-X[x,:])
        end
        return D
    end

    function update_means(X,means)
        # calculate Distance Matrix
        D = distance_matrix(X,means)
        # calculate new means
        T = zeros(K,p+1)
        for I in argmin(D, dims=2)
            let i = I[1], k = I[2]
                T[k,1:p] += X[i,:]
                T[k,p+1] += 1
            end
        end
        for k in 1:K
            means[k,:] = T[k,1:p]./T[k,p+1]
        end
        means
    end

    function clusters(X,means)
        D = distance_matrix(X,means)
        C = zeros(size(D))
        for I in argmin(D, dims=2)
            C[I] = I[1]
        end
        return C
    end

    # begin k-means
    N = size(x,1)
    p = size(x,2)
    means = x[ceil.(Int, rand(K)*N),:]

    for _ in 1:epochs
        update_means(x,means)
    end

    #plt[:imshow](means)

    C = clusters(x, means)[:,1]
    I = Int.(C[C .> 0.0])
    D = clusters(x, means)[:,2]
    J = Int.(D[D .> 0.0])
    return x[I,:] , x[J,:]
end

function generate_data(;n=100,p=50)
    #x = 1.0*(rand(n,p) .> 0.9)
    #x[1:Int(n/2),1:5] .= 1
    #x[(Int(n/2)+1):end,6:10] .= 1
    x = 1.0*(rand(n,p) .> 0.9)
    z = fill(0.0,n,10)
    z[1:Int(n/2),1:5] .= 1
    z[(Int(n/2)+1):end,6:10] .= 1
    z = 1.0*(rand(n,10) .<= z*0.3)
    x[:,1:10] = 1.0*((x[:,1:10] .+ z) .> 0)
    x
end

# function generate_data_bernoulli(;n=100,p=50)

Dopt = ADAM(params(gan.d), eta)
Gopt = ADAM(params(gan.g), eta)


function train1(gan, epochs::Int, batch_size::Int)
    for epoch=1:epochs
        loss_epoch = [0, 0, 0]
        @info "Epoch $epoch"
        for I in random_batch_index(x_st, batch_size)
            m = length(I)
            # println(m)
            # println(size(x[I,:]))
            # sample data and generate codes
            x_µ = Statistics.mean(x_st[I,:], dims=1)
            # println(size(x_µ))
            z = getcode(gan, m)
            # println(size(z))
            # x_z = gan.gg(z)

            # println(size(x_z))

            # x_z_µ = Statistics.mean(x_z, dims=2)

            # println(size(x_z_µ))
            # println(size(repeat(x_z_µ,m,1)))
            # discriminator training

            # println(size(hcat(x[I,:], repeat(x_µ,m,1))'))
            # println(size(vcat(x_z, repeat(x_z_µ,1,m))))
            # Dl = Dloss(gan, hcat(x[I,:], repeat(x_µ,m,1))', vcat(x_z, repeat(x_z_µ,1,m)))
            Dl = Dloss(gan, (x_st[I,:])', z)
            # println(Dl)

            # println(typeof((x[I,:])'))

            for i = 1:1
                psD = Flux.params(gan.d)
                # println(Flux.params(gan.g))
                gs = gradient(psD) do
                    training_loss = Dloss(gan, (x_st[I,:])', z)
                    # Insert what ever code you want here that needs Training loss, e.g. logging
                    return training_loss
                end
               # insert what ever code you want here that needs gradient
               # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge
                opt = ADAM(0.00001)
                Flux.Optimise.update!(opt, psD, gs)
                # println("janam?")
                # println(Flux.params(gan.g))

                #learningrate for 2d was 0.0001
            end

            # generator training

            Gl = Gloss(gan, z)

            for i = 1:1
                # println("yes")
                psG = Flux.params(gan.g)
                gs = gradient(psG) do
                    training_loss = Gloss(gan, z)
                    # Insert what ever code you want here that needs Training loss, e.g. logging
                    return training_loss
                end
               # insert what ever code you want here that needs gradient
               # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge
                opt = ADAM(0.00001)
                Flux.Optimise.update!(opt, psG, gs)
                # println("janam2?")
            end


            loss_epoch += [freeze(Dl), freeze(Gl), 1]
        end
        loss_epoch ./= loss_epoch[3]
        push!(loss[1],loss_epoch[1])
        push!(loss[2],loss_epoch[2])
        push!(gen_error,eval_gen(gan))
    end
end


function loss_plot(loss_array)
    plt = Gadfly.plot(x =[Float64(i) for i in 1:length(loss_array)], y = loss_array,  Geom.line)
    display(plt)
end



function activation_gen(x)
    # println(x)
    temp = fill(0.0, size(x)[1], size(x)[2])
    output = Zygote.Buffer(temp)
    # println(size(output))

    for j = 1:size(x)[1]
        if dataTypeArray[j] =="Continuous"
             output[j,:] = x[j,:]
             # println("inja")
        else
            # for i = 1: size(x)[2]
            #     # if (σ(x[j,i]) > 0.5)
            #     if (tanh(x[j,i]) > 0)
            #         output[j,i] = 1
            #     else
            #         output[j,i] = 0
            #     end
            # end
            output[j,:] = σ.(x[j,:])
        end
    end

    # println()

    copy(output)
end




function histogram_matrix(x, x_retr, title)
    (nobs, nvars) = size(x)

    column = floor(Int, sqrt(nvars))
    # (fig, ax) = subplots(4, floor(Int64,(nvars/4))+1, figsize=(8,8))

    (fig, ax) = subplots(ceil(Int64,(nvars/column)), column, figsize=(8,8))
    subplots_adjust(hspace=0.05, wspace=0.05)

    for i = 1:ceil(Int64,(nvars/column))
        for j = 1:column
            if (i-1)*column +j <= p
                # println((i-1)*4 +j)

                if dataTypeArray[((i-1)*column +j)] == "Binary"

                    ax[i,j][:hist](x[:,((i-1)*column+j)],  bins = 4, color = "#0f87bf", alpha = 0.5, label = "Original Data")#, "ob",mfc="none",
                    ax[i,j][:hist](x_retr[:,((i-1)*column+j)], bins = 4, color = "#ed1010", alpha = 0.5, label = "Our Method") #,"ob",mfc="none",
                else

                    ax[i,j][:hist](x[:,((i-1)*column+j)],  bins = 200, color = "#0f87bf", alpha = 0.5, label = "Original Data")#, "ob",mfc="none",
                    ax[i,j][:hist](x_retr[:,((i-1)*column+j)], bins = 200, color = "#ed1010", alpha = 0.5, label = "Our Method") #,"ob",mfc="none",
                end

                ax[i,j][:xaxis][:set_visible](false)
                ax[i,j][:yaxis][:set_visible](false)

            else
                ax[i,j][:xaxis][:set_visible](false)
                ax[i,j][:yaxis][:set_visible](false)
            end
        end
    end

    # Set tick positions
    for i = 1:ceil(Int64,(nvars/column))
        ax[i,1][:yaxis][:set_ticks_position]("left")
        ax[i,end][:yaxis][:set_ticks_position]("right")

    end

    for i=1:column
        ax[1,i][:xaxis][:set_ticks_position]("top")
        ax[end,i][:xaxis][:set_ticks_position]("bottom")

    end

    # red_patch = mpatches.Patch(color="#0f87bf", label='Our Method')
    # blue_patch = mpatches.Patch(color="#0f87bf", label='The Original Data')
    # green_patch = mpatches.Patch(color="#0f87bf", label='Gaussian Copula')
    fig.legend(["The Original Data", "Our Method"], loc="lower right")

    fig.suptitle(title, fontsize=16)
    display(fig)
end



function histogram_dimensions(x, x_retr, title)


    if floor(Int64, p/3) + 1 > 3
        rowSize = floor(Int64, p/3) + 1
        columnSize = 3
        set_default_plot_size(50cm, 50cm)
    else
        rowSize = p
        columnSize = 1
        set_default_plot_size(50cm, 20cm)
    end



    plotArray = fill(Gadfly.plot(), rowSize ,3)
    cnt = 1

    for i = 1: rowSize
        if cnt>p
            break
        end
        for j = 1: columnSize
            if cnt>p
                break
            end
            if (dataTypeArray[cnt] == "Binary")
                plt = Gadfly.plot(layer( x =x[:,cnt],  color=[colorant"black"], Geom.density(bandwidth = 0.2)),
                layer(x= x_retr[:,cnt],  color=[colorant"red"], Geom.density(bandwidth = 0.2)), Guide.manual_color_key("",["Original Data", "Synthetic Data"], ["black", "red"]),  Guide.xlabel("Dimension $cnt "))
            else
                plt = Gadfly.plot(layer( x =x[:,cnt],  color=[colorant"black"], Geom.density),
                layer(x= x_retr[:,cnt],  color=[colorant"red"], Geom.density), Guide.manual_color_key("",["Original Data", "Synthetic Data"], ["black", "red"]),  Guide.xlabel("Dimension $cnt "))
            end

            plotArray[i, j] = plt
            cnt+=1

        end


    end


    Gadfly.title(gridstack(plotArray), title)
    # Theme(key_title_font_size = 16pt))
end


function save_plot(plt, title)
    img = SVG(title)

    Gadfly.draw(img, plt)
    # Compose.load(title)
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
