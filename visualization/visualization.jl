#######################################Visualization############################################

################## scatteplot for every two variables ##################
function display_data_Gadfly(x, title)
    n = size(x)[1]
    p = size(x)[2]
    for i =1:p
        for j =i+1:p
            println("x$i, x$j")
            display(Gadfly.plot(x= x[:,i], y = x[:,j], Scale.x_continuous(minvalue=-5, maxvalue=5), Scale.y_continuous(minvalue=-5, maxvalue=5), Theme(point_size=4pt), Guide.title(title)))
        end
    end
end

################## 2-layer one-dimentsion histogram from original and synthetic data ##################
function histogram_plot(x1, x2, title)
    plt = Gadfly.plot(layer(x = x1, color=[colorant"black"], Geom.density),
        layer(x = x2, color=[colorant"red"], Geom.density), 
        Scale.x_continuous(minvalue=-1, maxvalue=1), Guide.title(title))
    display(plt)
end

################## optimization loss plot ##################
function loss_plot(loss_array)
    plt = Gadfly.plot(x =[Float64(i) for i in 1:length(loss_array)], y = loss_array,  Geom.line)
    display(plt)
end


##################### save plot function #####################
function save_plot(plt, title)
    img = SVG(title , 50cm, 50cm)
    Gadfly.draw(img, plt)
    # Compose.load(title)
end


################## scatterplot matrix ##################
function scatterplot_matrix(x, title)
    (nobs, nvars) = size(x)
    (fig, ax) = subplots(nvars, nvars, figsize=(8,8))
    subplots_adjust(hspace=0.05, wspace=0.05)
    # Plot data
    for i = 1:nvars
        for j = 1:nvars
            if j > i
                ax[i,j][:plot](x[:,j],x[:,i], markersize = 0.2,"ob",mfc="none")
            elseif i==j
                ax[i,j][:hist](x[:,i], bins=50)
            end
            ax[i,j][:xaxis][:set_visible](false)
            ax[i,j][:yaxis][:set_visible](false)
        end
    end

    # Set tick positions
    for i = 1:nvars
        ax[i,1][:yaxis][:set_ticks_position]("left")
        ax[i,end][:yaxis][:set_ticks_position]("right")
        ax[1,i][:xaxis][:set_ticks_position]("top")
        ax[end,i][:xaxis][:set_ticks_position]("bottom")
    end

    # Turn ticks on
    cc = repeat([nvars, 1],floor(Int, ceil(nvars/2)))
    for i = 1:nvars
        ax[i,cc[i]][:yaxis][:set_visible](true)
        ax[cc[i],i][:xaxis][:set_visible](true)
        # println(typeof(fig))
    end
    fig.suptitle(title, fontsize=16)
    display(fig)
end


################## histogram for all variables ##################
function histogram_matrix(x, x_retr, title)
    (nobs, nvars) = size(x)

    column = floor(Int, sqrt(nvars))
 
    (fig, ax) = subplots(ceil(Int64,(nvars/column)), column, figsize=(8,8))
    subplots_adjust(hspace=0.05, wspace=0.05)

    for i = 1:ceil(Int64,(nvars/column))
        for j = 1:column
            if (i-1)*column +j <= p

                if dataTypeArray[((i-1)*column +j)] == "Binary"
                    ax[i,j][:hist](x[:,((i-1)*column+j)],  bins = 4, color = "#0f87bf", alpha = 0.5, label = "Original Data")#, "ob",mfc="none",
                    ax[i,j][:hist](x_retr[:,((i-1)*column+j)], bins = 4, color = "#ed1010", alpha = 0.5, label = "Our Method") #,"ob",mfc="none",
                else
                    ax[i,j][:hist](x[:,((i-1)*column+j)],  bins = 50, color = "#0f87bf", alpha = 0.5, label = "Original Data")#, "ob",mfc="none",
                    ax[i,j][:hist](x_retr[:,((i-1)*column+j)], bins = 50, color = "#ed1010", alpha = 0.5, label = "Our Method") #,"ob",mfc="none",
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


################## density diagram for all variables ##################

function histogram_all_dimensions(x, syn, title)
    set_default_plot_size(50cm, 50cm)
    rowSize = floor(Int64, p/3) + 1

    plotArray = fill(Gadfly.plot(), rowSize ,3)
    cnt = 1

    for i = 1: rowSize
        if cnt>p
            break
        end
        for j = 1: 3
            if cnt>p
                break
            end
            plt = Gadfly.plot(
            layer( x =x[:,cnt], color=[colorant"black"], Theme(line_width=0.5mm, line_style=[:dot]), Geom.density(bandwidth = 0.6)), #(bandwidth = 0.2)
            layer(x= syn[:,cnt],  color=[colorant"red"], Theme(line_width=0.5mm,line_style=[:dot]), Geom.density(bandwidth = 0.6)),
            Guide.manual_color_key("Legend",["Original Data", "Synthetic Data"], [colorant"black", colorant"red"]),  Guide.xlabel("Dimension $cnt "))
            plotArray[i, j] = plt
            cnt+=1
        end
    end   
    Gadfly.title(gridstack(plotArray), title)
end