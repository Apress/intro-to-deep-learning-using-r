#Introduction to Deep Learning 
#Chapter 11 Examples 
#Taweh Beysolow II 
require(h2o)
require(pROC)
require(quantmod)
require(MASS)
require(LiblineaR)
require(rpart)
require(mlbench)
require(caret)
require(lmridge)
require(e1071)
require(Metrics)
require(Amelia)
require(randomForest)
require(FSelector)
require(RSNNS)
require(ggplot2)
require(lattice)
require(nnet)
require(pROC)
require(ROCR)
require(Amelia)
require(varSelRF)
require(caret)
require(mclust)
require(mxnet)
require(NLP)

#Clear the workspace 
rm(list = ls())

#Function to be used later
summaryStatistics <- function(array){
  Mean <- mean(array)
  Std <- sd(array)
  Min <- min(array)
  Max <- max(array)
  Range <- Max - Min 
  output <- data.frame("Mean" = Mean, "Std Dev" = Std, "Min" =  Min,"Max" = Max, "Range" = Range)
  return(output)
}


#Upload the necessary data 
data  <- read.csv("/Users/tawehbeysolow/Desktop/projectportfolio/SpeedDating.csv", header = TRUE, stringsAsFactors = TRUE)

################################################################################################
#Autoencoders 
###############################################################################################
#Data Preprocessing
#Creating response label 
second_date  <- matrix(nrow = nrow(data), ncol = 1)

for (i in 1:nrow(data)){
  if (data[i,1] + data[i,2] == 2){
    second_date[i]  <- 1
  } else {
    second_date[i]  <- 0
  }
}

#Creating new data set 
data  <- cbind(second_date, data)

################################################################################################
#Cleaning Data & Feature Transformation 

#Finding NA Observations
lappend <- function (lst, ...){
  lst <- c(lst, list(...))
  return(lst)
}

na_index <- list()
for (i in 1:ncol(data)){
  na_index <- lappend(na_index, which(is.na(data[,i])))
}

#Imputing NA Values where they are missing using EM Algorithm
#Step 1: Label Encoding Factor Variables to prepare for input to EM Algorithm
data$RaceM <- as.numeric(data$RaceM)
data$RaceF <- as.numeric(data$RaceF)

#Step 2: Inputting data to EM Algorithm 
data <-  amelia(x = data, m = 1,  boot.type = "none")$imputations$imp1

#Proof of EM Imputation 
na_index <- list()
for (i in 1:ncol(data)){
  na_index <- lappend(na_index, which(is.na(data[,i])))
}
na_index <- matrix(na_index, ncol = length(na_index), nrow = 1)
print(na_index)

#Scaling Features
data$AgeM <- scale(data$AgeM)
data$AgeF <- scale(data$AgeF)

################################################################################################
#Feature Selection 
corr <- cor(data)

#Converting all Columns to Numeric prior to Input 
for (i in 1:ncol(data)){
  data[,i] <- as.integer(data[,i])
}

#Random Forest Feature Selection Based on Importance of Classification 
data$second_date <- as.factor(data$second_date)
featImport <- random.forest.importance(second_date ~., data = data, importance.type = 1)
columns <- cutoff.k.percent(featImport, 0.4)
print(columns)

################################################################################################
#Model Training 
processedData <- data[, columns]

#Bayes Classifier 
AUC <- c()
for (i in 1:100){
  rows <- sample(1:nrow(processedData), 92)
  bayesClass <- naiveBayes(y = as.factor(second_date[rows]), x = processedData[rows, ], data = processedData)
  y_h <- predict(bayesClass, processedData[rows, ], type = c("class"))
  AUC <- append(roc(y_h, as.numeric(second_date[rows]))$auc, AUC)
}

summaryStatistics(AUC)

curve <- roc(y_h, as.numeric(second_date[rows]))
roc(y_h, as.numeric(second_date[rows]))
plot(curve, main = "Bayesian Classifier ROC")

################################################################################################
#Autoencoder
h2o.init()

training_data <- as.h2o(processedData, destination_frame = "train_data")

autoencoder <- h2o.deeplearning(x = colnames(processedData), 
                               training_frame = training_data, autoencoder = TRUE, activation = "Tanh",
                               hidden = c(6,5,6), epochs = 10)

autoencoder

#Reconstruct Original Data Set 
syntheticData <- h2o.anomaly(autoencoder, training_data, per_feature = FALSE)
errorRate <- as.data.frame(syntheticData)

#Plotting Error Rate of Feature Reconstruction
plot(sort(errorRate$Reconstruction.MSE), main = "Reconstruction Error Rate")


################################################################################################
#Removing Anomolies from Data 
train_data <- processedData[errorRate$Reconstruction.MSE < 0.01, ]

#Bayes Classifier 
AUC <- c()
for (i in 1:100){
  rows <- sample(1:nrow(processedData), 92)
  bayesClass1 <- naiveBayes(y = as.factor(second_date[rows]), x = processedData[rows, ], data = processedData)
  y_h1 <- predict(bayesClass1, processedData[rows, ], type = c("class"))
  AUC <- append(roc(y_h1, as.numeric(second_date[rows]))$auc, AUC)
}

#Summary Statistics
summaryStatistics(AUC)


###############################################################################################
#Using only Anomalies in Data Set
train_data <- processedData[errorRate$Reconstruction.MSE >= 0.01, ]

#Bayes Classifier 
AUC <- c()
for (i in 1:100){
  rows <- sample(1:nrow(processedData), 92)
  bayesClass2 <- naiveBayes(y = as.factor(second_date[rows]), x = processedData[rows, ], data = processedData)
  y_h2 <- predict(bayesClass2, processedData[rows, ], type = c("class"))
  AUC <- append(roc(y_h2, as.numeric(second_date[rows]))$auc, AUC)
}

#Summary Statistics 
summaryStatistics(AUC)

###############################################################################################
#Fitted Models and Out of Sample Performance

AUC1 <- AUC2 <- c()
for (i in 1:100){
  
  rows <- sample(1:nrow(processedData), 92)
  y_h1 <- predict(bayesClass1, processedData[-rows,], type = c("class"))
  y_h2 <- predict(bayesClass2, processedData[-rows,], type = c("class"))
  AUC1 <- append(roc(y_h1, as.numeric(second_date[-rows]))$auc, AUC1)
  AUC2 <- append(roc(y_h2, as.numeric(second_date[-rows]))$auc, AUC2)
}

summaryStatistics(AUC1)
summaryStatistics(AUC2)


#Non Anomaly Model
roc(y_h1, as.numeric(second_date[-rows]))
curve <- roc(y_h1, as.numeric(second_date[-rows]))
plot(curve, main = "Bayes Model 1 ROC Curve")

#Anomaly model
roc(y_h2, as.numeric(second_date[-rows]))
curve <- roc(y_h2, as.numeric(second_date[-rows]))
plot(curve, main = "Bayes Model 2 ROC Curve")


#Two Sided Hypothesis Test
require(BSDA)

z.test(x = AUC1, y = AUC2, alternative = "two.sided", mu = mean(AUC2) - mean(AUC1),
                 conf.level = 0.99, sigma.x = sd(AUC1), sigma.y = sd(AUC2))


################################################################################################
#Collaborative Filtering Example 
require(lsa)
require(bcv)
require(gdata)
require(Matrix)

#Clear the workspace 
rm(list = ls())

#Upload the data set 
#Please be patient this may take a handful of seconds to load. 
data <- read.xls("/Users/tawehbeysolow/Downloads/jester-data-3.xls", sheet = 1)
colnames(data) <- seq(1, ncol(data), 1)

#Exploring Data
head(data[,2:10])

#Converting 99s to NA Values
data[data == 99] <- NA

#Creating DataFrame to Be Used Later
origData <- data

#Converting NA Values to Column Means
for (i in 1:ncol(data)){
  data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
}

#Imputing Data via SVD
newData <- impute.svd(data, k = qr(data)$rank, tol = 1e-4, maxiter = 200)
print(newData$rss)
head(data[, 2:10])


#Instantiating Empty Matrix to place cosine distances in
itemData <- matrix(NA, nrow = ncol(data), ncol = 11,
                   dimnames=list(colnames(data)))

#Getting Cosine Distances 
for (i in 1:nrow(itemData)){
  for (j in 1:ncol(itemData)){
    itemData[i,j] <- cosine(data[,i], data[,j])
  }
}

head(itemData[, 1:10])

#Creating Matrix for ranking similarities
similarMat <- matrix(NA, nrow = ncol(itemData), ncol = 11)

#Sorting Data Within Item Data Matrix
for(i in 1:ncol(itemData)) {
  rows <- order(itemData[,i], decreasing = TRUE)
  similarMat[i,] <- (t(head(n=11, rownames(data[rows ,][i]))))
}

#Printing Result
similarMat





