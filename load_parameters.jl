
function load_parameter(data_string)
    if data_string == "sim" 
        path = "C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\Simulation Design data\\Our Method\\Parameters\\"
    elseif data_string == "ist" 
        path = "C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\IST data\\Our Method\\Parameters\\"
    elseif data_string =="toy"
        path = "C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\toyData\\Our Method\\Parameters\\"
    end

    shiftArray = convert(Array, CSV.read(joinpath(path, "ShiftArray.csv"), header=false))
    lambdaArray = convert(Array, CSV.read(joinpath(path,"lambdaArray.csv", header=false))
    alphaArray = convert(Array, CSV.read(joinpath(path,"alphaArray.csv", header=false))
    peak1Array = convert(Array, CSV.read(joinpath(path,"peak1Array.csv", header=false))
    peak2Array = convert(Array, CSV.read(joinpath(path,"peak2Array.csv", header=false))
    powerArray = convert(Array, CSV.read(joinpath(path,"powerArray.csv", header=false))

    return shiftArray, lambdaArray, alphaArray, peak1Array, peak2Array, powerArray

    # shiftArray, lambdaArray, alphaArray, peak1Array, peak2Array, powerArray = load_parameter(data_string)
end