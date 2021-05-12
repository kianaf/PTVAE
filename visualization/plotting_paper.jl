######################################################################################################################
####### Figure 2 #####################################################################################################
######################################################################################################################
function toy_data_load!()
    x = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\toyData\\Our Method\\Original Data\\original_data.csv", header=false))
    syndata_prior = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\toyData\\Our Method\\Synthetic Data\\syndata_prior.csv", header=false))
    syndata_prior_standard = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\toyData\\Our Method\\Synthetic Data\\syndata_prior_standard.csv", header=false))
    x_tr_logit = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\toyData\\Our Method\\Transformed Data\\x_tr_logit.csv", header=false))
    
    return x, syndata_prior, syndata_prior_standard, x_tr_logit
end


function plot_for_PTVAE()
    (nobs, nvars) = size(x)
    # (fig, ax) = subplots(4, 3, gridspec_kw={"width_ratios": [4, 3]})#figsize=(8,8))
    (fig, ax) = subplots(4, 3, figsize = (6,6))
    PyPlot.subplots_adjust(bottom=0.1, right=0.8, top=0.9)


    # x, syndata_prior, syndata_prior_standard = toy_data_load!()
    subplots_adjust(hspace=0.2, wspace=0.2)

    # dataList = [x, round_discrete(syndata_prior_standard), x_tr_logit, round_discrete(syndata_prior)]
    # titleList = ["Original data", "Synthetic data (VAE)", "Transformed data", "Synthetic data (PTVAE)"]

    dataList = [x, x_tr_logit_bimodality, x_tr_logit_IQR, round_discrete(syndata_prior)]
    titleList = ["Original data", "Synthetic data (VAE)", "Transformed data", "Synthetic data (PTVAE)"]

    # Plot data
    for i = 1:4
        data = dataList[i]
        for j = 1:3
            if j == 1
                ax[i,j][:plot](data[:,2],data[:,1],  markersize = 1.5, "ob",mew = 0.1,mec="black", mfc="lightblue") # ,
                ax[i,j][:xaxis][:set_visible](true)
                ax[i,j][:yaxis][:set_visible](true)
                # println([floor(minimum(data[:,2])), ceil(maximum(data[:,2])),floor(minimum(data[:,1])), ceil(maximum(data[:,1]))])
                # if i==3
                #     ax[i,j].set(xlim=(-3.5, 2.5) ,ylim = (-3.5, 2.5), autoscale_on = false)
                # else
                    ax[i,j].set(xlim=(floor(minimum(data[:,2])), ceil(maximum(data[:,2]))) ,ylim = (floor(minimum(data[:,1])), ceil(maximum(data[:,1]))), autoscale_on = false)
                # end
                ax[i,j].set_yticks(range(floor(minimum(data[:,1])), ceil(maximum(data[:,1])), step = 0.5))

                
                ax[i,j].xaxis.set_tick_params(length=2,width=0.5,labelsize=4)
                ax[i,j].yaxis.set_tick_params(length=2,width=0.5,labelsize=4)
                # ax[i,j].tick_params(labelsize=8)
                if i==1
                    ax[i,j].set_title("Bivariate distribution", fontsize=5)
                end
                ax[i,j].set_ylabel(titleList[i], fontsize=5)
                # ax[i,j].yaxis.set_major_formatter
            elseif j == 2
                ax[i,j][:hist](data[:,2], bins=30, color = "lightblue", edgecolor = "black",linewidth = 0.4)
                ax[i,j][:xaxis][:set_visible](true)
                ax[i,j][:yaxis][:set_visible](false)
                ax[i,j].set(xlim=(floor(minimum(data[:,2])), ceil(maximum(data[:,2]))) , autoscale_on = false)
                
                ax[i,j].xaxis.set_tick_params(length=2,width=0.5,labelsize=4)
                ax[i,j].yaxis.set_tick_params(length=2,width=0.5,labelsize=4)
                # ax[i,j].xaxis([floor(minimum(data[:,2])), ceil(maximum(data[:,2]))])
                if i ==1
                    ax[i,j].set_title("Bimodal variable", fontsize=5)
                end
            else
                ax[i,j][:hist](data[:,1], bins=30, color = "lightblue", edgecolor = "black",linewidth = 0.4)
                ax[i,j][:xaxis][:set_visible](true)
                ax[i,j][:yaxis][:set_visible](false)
                ax[i,j].set(xlim=(floor(minimum(data[:,1])), ceil(maximum(data[:,1]))) , autoscale_on = false)

                ax[i,j].xaxis.set_tick_params(length=2,width=0.5,labelsize=4)
                ax[i,j].yaxis.set_tick_params(length=2,width=0.5,labelsize=4)
                # ax[i,j].xaxis([floor(minimum(data[:,2])), ceil(maximum(data[:,2]))])
                if i ==1
                    ax[i,j].set_title("Skewed variable", fontsize=5)
                end
            end            
        end
    end


    fig.suptitle("", fontsize=16)
    display(fig)
end



######################################################################################################################
####### Figure 3 #####################################################################################################
######################################################################################################################
function toy_data_different_transformation_load!()
    x_tr_logit_bimodality5 = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\toyData\\Our Method\\Transformed Data\\x_tr_logit_bimodality5.csv", header=false))
    x_tr_logit_bimodality10 = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\toyData\\Our Method\\Transformed Data\\x_tr_logit_bimodality5.csv", header=false))
    x = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\toyData\\Our Method\\Original Data\\original_data.csv", header=false))
    x_tr_logit_IQR = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\toyData\\Our Method\\Transformed Data\\x_tr_logit.csv", header=false))
    x_tr_logit_kl5= convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\toyData\\Our Method\\Transformed Data\\x_tr_logit_kl5.csv", header=false))
    x_tr_logit_MLE5 = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\toyData\\Our Method\\Transformed Data\\x_tr_logit_MLE5.csv", header=false))

    
    return x_tr_logit_bimodality5, x_tr_logit_bimodality10, x, x_tr_logit_IQR,  x_tr_logit_kl5, x_tr_logit_MLE5
    # x_tr_logit_bimodality5, x_tr_logit_bimodality10, x, x_tr_logit_IQR,  x_tr_logit_kl5, x_tr_logit_MLE5 = toy_data_different_transformation_load!()
end


function bimodality_transform_criteria_plot()
    (nobs, nvars) = size(x)
    # (fig, ax) = subplots(4, 3, gridspec_kw={"width_ratios": [4, 3]})#figsize=(8,8))
    (fig, ax) = subplots(4, 3, figsize = (6,6))
    PyPlot.subplots_adjust(bottom=0.1, right=0.8, top=0.9)


    # x, syndata_prior, syndata_prior_standard = toy_data_load!()
    subplots_adjust(hspace=0.2, wspace=0.2)

    # dataList = [x, round_discrete(syndata_prior_standard), x_tr_logit, round_discrete(syndata_prior)]
    # titleList = ["Original data", "Synthetic data (VAE)", "Transformed data", "Synthetic data (PTVAE)"]

    dataList = [x, x_tr_logit_IQR, x_tr_logit_bimodality5, x_tr_logit_MLE5]
    titleList = ["Original data", "2-sigma rule", "Bimodality coefficient", "Maximum likelihood"]

    # Plot data
    for i = 1:4
        data = dataList[i]
        for j = 1:3
            if j == 1
                ax[i,j][:plot](data[:,2],data[:,1],  markersize = 1.5, "ob",mew = 0.1,mec="black", mfc="lightblue") # ,
                ax[i,j][:xaxis][:set_visible](true)
                ax[i,j][:yaxis][:set_visible](true)
                ax[i,j].set(xlim=(floor(minimum(data[:,2])), ceil(maximum(data[:,2]))) ,ylim = (floor(minimum(data[:,1])), ceil(maximum(data[:,1]))), autoscale_on = false)
                ax[i,j].set_yticks(range(floor(minimum(data[:,1])), ceil(maximum(data[:,1])), step = 0.5))

                
                ax[i,j].xaxis.set_tick_params(length=2,width=0.5,labelsize=4)
                ax[i,j].yaxis.set_tick_params(length=2,width=0.5,labelsize=4)
                if i==1
                    ax[i,j].set_title("Bivariate distribution", fontsize=5)
                end
                ax[i,j].set_ylabel(titleList[i], fontsize=5)
            elseif j == 2
                ax[i,j][:hist](data[:,2], bins=30, color = "lightblue", edgecolor = "black",linewidth = 0.4)
                ax[i,j][:xaxis][:set_visible](true)
                ax[i,j][:yaxis][:set_visible](false)
                ax[i,j].set(xlim=(floor(minimum(data[:,2])), ceil(maximum(data[:,2]))) , autoscale_on = false)
                
                ax[i,j].xaxis.set_tick_params(length=2,width=0.5,labelsize=4)
                ax[i,j].yaxis.set_tick_params(length=2,width=0.5,labelsize=4)
                if i ==1
                    ax[i,j].set_title("Bimodal variable", fontsize=5)
                end
            else
                ax[i,j][:hist](data[:,1], bins=30, color = "lightblue", edgecolor = "black",linewidth = 0.4)
                ax[i,j][:xaxis][:set_visible](true)
                ax[i,j][:yaxis][:set_visible](false)
                ax[i,j].set(xlim=(floor(minimum(data[:,1])), ceil(maximum(data[:,1]))) , autoscale_on = false)

                ax[i,j].xaxis.set_tick_params(length=2,width=0.5,labelsize=4)
                ax[i,j].yaxis.set_tick_params(length=2,width=0.5,labelsize=4)
                if i ==1
                    ax[i,j].set_title("Skewed variable", fontsize=5)
                end
            end            
        end
    end


    fig.suptitle("", fontsize=16)
    display(fig)
end

######################################################################################################################
####### Figure 4 #####################################################################################################
######################################################################################################################


function load_all_methods_sim!()
    ourMethod = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\Simulation Design data\\Our Method\\Synthetic Data\\syndata_prior.csv", header=false))
    FedSyn = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\Simulation Design data\\Norta\\Norta-J\\FedSyn_method4.csv", header=false))
    VAE = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\Simulation Design data\\VAE\\Synthetic Data\\syndata_prior.csv", header=false))
    QVAE = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\Simulation Design data\\QVAE\\Synthetic Data\\syndata_prior.csv", header=false))
    GAN = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\Simulation Design data\\GAN\\Synthetic Data\\syndata.csv", header=false))
    return round_discrete(ourMethod),round_discrete(FedSyn), round_discrete(VAE), round_discrete(QVAE), round_discrete(GAN)
end

function density_matrix_sim(x, ourMethod, FedSyn, VAE, QVAE, GAN, title)
    (fig, ax) = subplots(5, 3, figsize=(8,8))
    subplots_adjust(hspace=0.4, wspace=0.3)
    PyPlot.subplots_adjust(bottom=0.1, right=0.6, top=0.9)
    dataList = [ourMethod, FedSyn, VAE, QVAE, GAN]
    variableList = ["Slightly skewed", "Severely skewed", "Bimodal"]
    titleList = ["PTVAE", "Norta-j", "Standard VAE", "QVAE", "GAN"]
    indexList = [6,8,20]
    for i = 1:5
        data = dataList[i]
        for j = 1:3
            binCal = ((maximum(x[:,indexList[j]]) - minimum(x[:,indexList[j]]))/50)
            println(binCal)
            data_x = sort(floor.(x[:,indexList[j]] ./binCal ) .* binCal)
            data_syn = sort(floor.(data[:,indexList[j]] ./ binCal) .*binCal)
            DictFrequency= StatsBase.countmap(data_x)
            frequency =[get(DictFrequency, data_x[k], 0) for k =1:n] 

            DictFrequencydata= StatsBase.countmap(data_syn)
            frequencydata =[get(DictFrequencydata,data_syn[k] , 0) for k =1:n] 
            
            ax[i,j][:plot](data_x, frequency, color = "black", alpha = 0.5, label = "Original Data", mec="black"),
            ax[i,j][:plot](data_syn, frequencydata, color = "#e73134", alpha = 0.5, label = "Our Method", mec="#e73134") #,"ob",mfc="none",
            if i==1
                ax[i,j].set_title(variableList[j], fontsize=5)
            end
            if j ==1
                ax[i,j].set_ylabel(titleList[i], fontsize=5)
            end
            ax[i,j].xaxis.set_tick_params(length=2,width=0.5,labelsize=4)
            ax[i,j].yaxis.set_tick_params(length=2,width=0.5,labelsize=4)
        end
    end
    fig.legend(["Original Data", "Synthetic Data"], loc="upper right",  bbox_to_anchor=(0.65, 0.90), fontsize= 5)
    
    fig.suptitle(title, fontsize=16)

    display(fig)
    savefig("C:\\Users\\farhadyar\\Desktop\\density_matrix_sim.pdf")

end

######################################################################################################################
####### Figure 5 #####################################################################################################
######################################################################################################################

function load_all_methods_ist!()
    ourMethod = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\IST data\\Our Method\\Synthetic Data\\syndata_prior.csv", header=false))
    FedSyn = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\IST data\\Norta\\Norta-J\\FedSynIST_method4.csv", header=false))
    VAE = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\IST data\\VAE\\Synthetic Data\\syndata_prior.csv", header=false))
    GAN = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Final Data alpha greater than zero\\IST data\\GAN\\Synthetic Data\\syndata.csv", header=false))
    return round_discrete(ourMethod), round_discrete(FedSyn),round_discrete(VAE), round_discrete(GAN)
end


function density_matrix_ist(x, ourMethod, FedSyn, VAE, GAN, title)
    (fig, ax) = subplots(4, 2, figsize=(8,8))
    subplots_adjust(hspace=0.4, wspace=0.3)
    PyPlot.subplots_adjust(bottom=0.1, right=0.6, top=0.9)
    dataList = [ourMethod, FedSyn, VAE, GAN]
    variableList = ["Age", "Blood Pressure"]
    titleList = ["PTVAE", "Norta-j", "Standard VAE", "GAN"]
    indexList = [4,8]
    for i = 1:4
        data = dataList[i]
        for j = 1:2
            if indexList[j] ==8
                data_x = sort(floor.(x[:,indexList[j]] ./10) .*10)
                data_syn = sort(floor.(data[:,indexList[j]] ./10) .*10)
            else
                data_x = sort(x[:,indexList[j]])
                data_syn = sort(data[:,indexList[j]])
            end
            DictFrequency= StatsBase.countmap(data_x)
            frequency =[get(DictFrequency, data_x[k], 0) for k =1:n] 

            DictFrequencydata= StatsBase.countmap(data_syn)
            frequencydata =[get(DictFrequencydata,data_syn[k] , 0) for k =1:n] 
            
            ax[i,j][:plot](data_x, frequency, color = "black", alpha = 0.5, label = "Original Data", mec="black"),
            ax[i,j][:plot](data_syn, frequencydata, color = "#e73134", alpha = 0.5, label = "Our Method", mec="#e73134") #,"ob",mfc="none",
            if i==1
                ax[i,j].set_title(variableList[j], fontsize=5)
            end
            if j ==1
                ax[i,j].set_ylabel(titleList[i], fontsize=5)
            end
            ax[i,j].xaxis.set_tick_params(length=2,width=0.5,labelsize=4)
            ax[i,j].yaxis.set_tick_params(length=2,width=0.5,labelsize=4)
        end
    end
    fig.legend(["Original Data", "Synthetic Data"], loc="upper right",  bbox_to_anchor=(0.65, 0.90), fontsize= 5)
    
    fig.suptitle(title, fontsize=16)
    display(fig)
    savefig("C:\\Users\\farhadyar\\Desktop\\density_matrix_ist.pdf")

end
