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


function generate_toy_example()
    Random.seed!(42)
    m = 20
    n = 5000
    p = 2
    x1 = initialize_skewed(p,n)
    x2 = initialize_bimodal(p,n)
    x = hcat(x1[:,1], x2[:,1])
    return m, n, p, x
end
