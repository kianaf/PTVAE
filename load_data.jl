function load_dataset(data_string)
    if data_string == "ist"
        
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
                cnt+=1
            end    
        end

        n = size(x)[1]

    elseif data_string == "sim"
        
        Random.seed!(11)
        n=2500
        p=21
        m = 50
        x = convert(Array, CSV.read("data\\simulation.csv", header=true))

        # data_withheader = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Simulation Design\\data_scenario1_Bimodal_Binary.csv", header=false))
        # header = data_withheader[1, 1:p]


    elseif data_string == "toy"
        includet("data\\toy_example_generation.jl")
        m, n, p, x = generate_toy_example()

        #or
        x = convert(Array, CSV.read("data\\toy_example.csv", header=true))
        m = 20
        n = 5000
        p = 2
    else
        # your own data
        x = convert(Array, CSV.read(data_string, header=true))
        n = size(x)[1]
        p = size(x)[2]
        m = 1 # customize it 
    end

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

    return m, n, p, x, dataTypeArray
end
