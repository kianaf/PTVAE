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