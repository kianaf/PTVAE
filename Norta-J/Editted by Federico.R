## gcipdr_IST_analysis.R contains R commands to execute 'gcipdr application to multi-center IST data' (from omonimous repository). 
## Copyright (C) 2019 Federico Bonofiglio

## This Program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This Program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with This Program.  If not, see <https://www.gnu.org/licenses/>. 

if (!require("devtools")) {
  install.packages("devtools")
  library(devtools)
}


## Install 'JohnsonDistribution' dependency (only available on CRAN archives)

url <- "https://cran.r-project.org/src/contrib/Archive/JohnsonDistribution/JohnsonDistribution_0.24.tar.gz"
pkgFile <- "JohnsonDistribution_0.24.tar.gz"
download.file(url = url, destfile = pkgFile)

install.packages(pkgs=pkgFile, type="source", repos=NULL)

unlink(pkgFile)


### INSTALL package (install other dependencies manually if needed)


R_REMOTES_NO_ERRORS_FROM_WARNINGS="true"
install_github("bonorico/gcipdr")


# load libraries
# trace("Simulate.many.datasets", edit = TRUE)


library(gcipdr)

if (!require("cowplot")) {
  install.packages("cowplot")
  library(cowplot)
}

if (!require("meta")) {
  install.packages("meta")
  library(meta)
}

if (!require("metafor")) {
  install.packages("metafor")
  library(metafor)
}

if (!require("lme4")) {
  install.packages("lme4")
  library(lme4)
}



## do not run !
## options("mc.cores") <- 3 ## set up your global 'mclapply()' forking options other than default (2 cores). Beware: changing default options might not reproduce results as shown in paper.


url2 <- "C:\\Users\\farhadyar\\Desktop\\simulationDesign.csv"

# url2 <- file.path(getwd(), "simulationDesign.csv")

x_simulation <- read.csv( url2 )



for(i in 1:21)
  x_simulation[,i] <- x_simulation[,i] - min(x_simulation[,i])


#make all the data positive
#x_simulation[,20] <- x_simulation[,20] - min(x_simulation[,20])


#######################  COMMENT (Bono): current example runs in approx 3 minutes but ...
options(mc.cores = 3L)  ## .. change accordingly to your Nr of cores (> 3) minus one, to speed up calculations  
######################

######### LOAD COPULA DATA ##########

seed <- 49632

    jiseed <- as.integer(paste(seed, 3, sep = ""))
    set.seed(jiseed, "L'Ecuyer") # delete this line to assess stability
               
    print(system.time(
        simul <-  artificial_data_object <- Simulate.many.datasets( list(x_simulation), NULL, 3,
                    checkdata = TRUE, tabulate.similar.data = TRUE, NI_maxEval = 2000 )[[1]]
    ))
    
  

### COMMENT (Bono): with current settings, copula simulation (Johnson marginals) runs ok. You'll see some warnings due to fitted correlation values exceeding tolerance treshold (see printed table "Badly generated correlations"). This occurs often, is not alarming. Also the highest error (column 'diff') is only about |0.1| which should not be worrying at all. Nevertheless, if you feel you might want to try better accuracy, just increase argument 'NI_maxEval' to maybe 5K or 10K. The further you go, the more expensive optimization is, and typiically with little added gain. Alternatively, you could swap to stochastic integration. But my feeling is that this would bring no big advantage here. I expect the above settings to yield OK results already.



### COMMENT (Bono): you do not need "PoolArtifData" function. This was just meant in my article to pool synthetic data from different centers, but this is not the case here, because you are only working with one single data set (right ?)

### COMMENT (Bono): to extract data just see 
names(simul)

## the synthetic data is in

syntheticData <- simul$similar.data

class(syntheticData)
length(syntheticData) # which consists of 100 synthetic realizations of original data ...


write.csv(syntheticData[[1]], "C:\\Users\\farhadyar\\Desktop\\FedSyn_method3.csv")
