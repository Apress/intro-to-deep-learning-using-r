#Chapter 6 Code Examples

#Clear the workspace 
rm(list = ls())

#Load the necessary packages 
require(darch)
require(RSNNS)


#Elman Network 
data(snnsData)
inputs <- snnsData$eight_016.pat[,inputColumns(snnsData$eight_016.pat)]
outputs <- snnsData$eight_016.pat[,outputColumns(snnsData$eight_016.pat)]

par(mfrow=c(1,2))

modelElman <- elman(inputs, outputs, size=8, learnFuncParams=c(0.1), maxit=1000)
modelElman
modelJordan <- jordan(inputs, outputs, size=8, learnFuncParams=c(0.1), maxit=1000)
modelJordan

plotIterativeError(modelElman)
plotIterativeError(modelJordan)

summary(modelElman)
summary(modelJordan)







#Loading Iris Data Set
data  <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                    sep = ",", header = FALSE)

x <- as.matrix(data[,1:4])

#Restricted Boltzmann Machine 
RBM <- rbm.train(x, hidden = 3, 
                 numepochs = 30, 
                 cd = 10, 
                 learningrate = 0.08)
RBM

