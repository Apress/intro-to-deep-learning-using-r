#Introduction To Deep Learning 
#Chapter 10: Applied Machine Learning 
#Example 10.1 Quantitative Finance: Asset Price Prediction 
#Clear the workspace
rm(list = ls())

#Upload the necessary packages 
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
require(class)


#Summary Statistics Function 
#We will use this later to evaluate our model performance 
summaryStatistics <- function(array){
  Mean <- mean(array)
  Std <- sd(array)
  Min <- min(array)
  Max <- max(array)
  Range <- Max - Min 
  output <- data.frame("Mean" = Mean, "Std Dev" = Std, "Min" =  Min,"Max" = Max, "Range" = Range)
  return(output)
}


################################################################################################
#Loading Data From Yahoo Finance
stocks <- c("F", "SPY", "DJIA", "HAL", "MSFT", "SWN", "SJM", "SLG", "STJ")
stockData  <- list()
for(i in stocks){
  stockData[[i]] <- getSymbols(i, src = 'yahoo', auto.assign = FALSE,  return.class = "xts", from = "2013-01-01", to = "2017-01-01")
}
#Creating Matrix of close prices 
df  <- matrix(nrow = nrow(stockData[[1]]), ncol = length(stockData))
for (i in 1:length(stockData)){
  df[,i]  <- stockData[[i]][,6]
}
#Calculating Returns
return_df  <- matrix(nrow = nrow(df), ncol = ncol(df))
for (j in 1:ncol(return_df)){
  for(i in 1:nrow(return_df) - 1){
    return_df[i,j]  <- (df[i+1, j]/df[i,j]) - 1
  }
}

################################################################################################
#Data Preprocessing
################################################################################################
#Feature Selection 
#Removing last row since it is an NA VALUE
return_df  <- return_df[-nrow(return_df), ]

corr_df <- cor(return_df[, -1])
colnames(corr_df) <- rownames(corr_df) <- stocks[-1]
corr_df
#Making DataFrame with all values except label IE all columns except for Ford since we are trying to predict this
#Determing Which Variables Are Unnecessary 
pca_df  <- return_df[, -1]
pca  <- prcomp(scale(pca_df))
summary(pca)

#Run Summary To Determin the Proportion of the Variance you are getting back 
#In this case, since PCs 7 and 8 contribute relatively little, we will exclude variables 
#7 and 8. I have said the cutoff is 1%, and although 7 barely meets this threshold, I think 6 variables will be enough 
#Editing Existing Return Data 
new_returns  <- return_df[, 1:7]
colnames(new_returns)  <- stocks[1:7]
head(new_returns)


################################################################################################
#Parameter Selection
################################################################################################
#Models To Be Chosen 
#Ridge Regression 
k <- sort(rnorm(100))
mse_ridge <- ridge_r2 <- c()
for (j in 1:length(k)){
    valid_rows <- sample(1:(nrow(return_df)/2))
    valid_set <- new_returns[valid_rows, -1]
    valid_y <- new_returns[valid_rows, 1]
    #Ridge Regression 
    ridgeReg <- lmridge(valid_y ~ valid_set[,1] + valid_set[,2] + valid_set[,3] + valid_set[,4] 
                             + valid_set[,5] + valid_set[,6], data = as.data.frame(valid_set), type = type,  K = k[j])
    mse_ridge <- append(rstats1.lmridge(ridgeReg)$mse, mse_ridge)
}


#Plots of MSE and R2 as Tuning Parameter Grows
plot(k, mse_ridge, main = "MSE over Tuning Parameter Size", xlab = "K", ylab = "MSE", type = "l", 
     col = "cadetblue")


################################################################################################
#Support Vector Regression 
#Kernel Selection
svr_mse <- svr_r2 <- c()
k <- c("linear", "polynomial", "sigmoid")
for (i in 1:length(k)){
  valid_rows <- sample(1:(nrow(return_df)/2))
  valid_set <- new_returns[valid_rows, -1]
  valid_y <- new_returns[valid_rows, 1]
  
  SVR <- svm(valid_y ~ valid_set[,1] + valid_set[,2] + valid_set[,3] + valid_set[,4] 
             + valid_set[,5] + valid_set[,6], kernel = k[i])
  svr_y <- predict(SVR, data = valid_set)
  svr_mse <- append(mse(valid_y, svr_y), svr_mse)
}

#Plots of MSE and R2 as Tuning Parameter Grows
plot(svr_mse, main = "MSE over Tuning Parameter Size", xlab = "K", ylab = "MSE", type = "l", 
       col = "cadetblue")

################################################################################################
#Predicting out of Sample with Tuned Models 

#Tuned Ridge Regression 
ridgeReg <- lmridge(valid_y ~ valid_set[,1] + valid_set[,2] + valid_set[,3] + valid_set[,4] 
                    + valid_set[,5] + valid_set[,6], data = as.data.frame(valid_set), type = type,  K = 1)

y_h <- predict(ridgeReg, as.data.frame(new_returns[-valid_rows, -1]))
mse_ridge <- mse(new_returns[-valid_rows, 1], y_h)


#Tuned Support Vector Regression 
svr <-   SVR <- svm(valid_y ~ valid_set[,1] + valid_set[,2] + valid_set[,3] + valid_set[,4] 
                    + valid_set[,5] + valid_set[,6], kernel = "polynomial")

svr_y <- predict(svr, data = new_returns[-valid_rows, -1])
svr_mse <- mse(new_returns[-valid_rows, 1], svr_y)

#Tail of Predicted Value DataFrames
svr_pred <- cbind(new_returns[-valid_rows, 1], svr_y)
colnames(svr_pred) <- c("Actual", "Predicted")
tail(svr_pred)

ridge_pred <- cbind(new_returns[-valid_rows, 1], y_h)
colnames(ridge_pred) <- c("Actual", "Predicted")
tail(ridge_pred)


cat("MSE for Ridge Regression: ", mse_ridge)
cat("MSE for Support Vector Regression: ", svr_mse)


################################################################################################
################################################################################################
#Classification Problem 
#Clear the workspace 
rm(list = ls())


#Summary Statistics Function 
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

#Scaling Age Features using Gaussian Normalization 
summaryStatistics(data$AgeM)
summaryStatistics(data$AgeF)

#Making Histograms of Data 
hist(data$AgeM, main = "Distribution of Age in Males", xlab = "Age", ylab = "Frequency", col = "darkorange3")

hist(data$AgeF, main = "Distribution of Age in Females", xlab = "Age", ylab = "Frequency", col = "firebrick1")



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

#############################
#Method 1: Logistic Regression 
lambda <- seq(.01, 1, .01)
AUC <- c()
for (i in 1:length(lambda)){
  rows <- sample(1:nrow(processedData), nrow(processedData)/2)
  logReg <- glm(as.factor(second_date[rows]) ~., data = processedData[rows, ], family = binomial(link = "logit"), method = "glm.fit")
  y_h <- ifelse(logReg$fitted.values >= lambda[i], 1, 0)
  AUC <- append(roc(y_h, as.numeric(second_date[-rows]))$auc, AUC)
}

#Summary Statistics and Various Plots
plot(lambda[-1], AUC, main = "AUC over Lambda Value \n(Logistic Regression)", 
     xlab = "Lambda", ylab = "AUC", type = "l", col = "cadetblue")

#Tuned Model
AUC <- c()
for (i in 1:length(lambda)){
  rows <- sample(1:nrow(processedData), nrow(processedData)/2)
  logReg <- glm(as.factor(second_date[rows]) ~., data = processedData[rows, ], family = binomial(link = "logit"), method = "glm.fit")
  y_h <- ifelse(logReg$fitted.values >= 0.15, 1, 0)
  AUC <- append(roc(y_h, as.numeric(second_date[-rows]))$auc, AUC)
}

#Summary Statistics and Various Plots
plot(AUC, main = "AUC over 100 Iterations \n(Logistic Regression, lambda = 0.15)", 
     xlab = "Iterations", ylab = "AUC", type = "l", col = "cadetblue")

hist(AUC, main = "Histogram for AUC \n(Logistic Regression, lambda = 0.15)", 
     xlab = "AUC Value", ylab = "Frequency", col = "firebrick3")

summaryStatistics(AUC)



#############################
#Method 2: Bayesian Classifier 
AUC <- c()
for (i in 1:100){
  rows <- sample(1:nrow(processedData), 92)
  bayesClass <- naiveBayes(y = as.factor(second_date[rows]), x = processedData[rows, ], data = processedData)
  y_h <- predict(bayesClass, processedData[rows, ], type = c("class"))
  AUC <- append(roc(y_h, as.numeric(second_date[rows]))$auc, AUC)

}

#Summary Statistics and Various Plots
plot(AUC, main = "AUC over 100 Iterations \n(Naive Bayes Classifier)", 
     xlab = "Iterations", ylab = "AUC", type = "l", col = "cadetblue")

hist(AUC, main = "Histogram for AUC \n(Naive Bayes Classifier)", 
     xlab = "AUC Value", ylab = "Frequency", col = "firebrick3")

summaryStatistics(AUC)


#Predicting out of Sample 
y_h <- predict(bayesClass, processedData[-rows, ], type = c("class"))
roc(y_h, as.numeric(second_date[-rows]))$auc


#############################
#Method 3: K-Nearest Neighbor
#Tuning K Parameter
K <- seq(1, 40, 1)
AUC <- c()
for (i in 1:length(K)){
  rows <- sample(1:nrow(processedData), nrow(processedData)/2)
  y_h <- knn(train = processedData[rows, ], test = processedData[rows,], cl = second_date[rows], k = K[i], use.all = TRUE)
  AUC <- append(roc(y_h, as.numeric(second_date[rows]))$auc, AUC)
}

#Summary Statistics and Various Plots
plot(AUC, main = "AUC over K Value \n(K Nearest Neighbor)", 
     xlab = "K", ylab = "AUC", type = "l", col = "cadetblue")

#Tuned Model 
AUC <- c()
for (i in 1:100){
  y_h <- knn(train = processedData[rows, ], test = processedData[-rows,], cl = second_date[rows], k = 3, use.all = TRUE)
  AUC <- append(roc(y_h, as.numeric(second_date[rows]))$auc, AUC)
}

#Summary Statistics and Various Plots
plot(AUC, main = "AUC over 100 Iterations \n(K Nearest Neighbor, K = 3)", 
     xlab = "Iterations", ylab = "AUC", type = "l", col = "cadetblue")

hist(AUC, main = "Histogram for AUC \n(K Nearest Neighbor, K = 3)", 
     xlab = "AUC Value", ylab = "Frequency", col = "firebrick3")


summaryStatistics(AUC)

#Predicting out of Sample 
y_h <- knn(train = processedData[rows, ], test = processedData[-rows, ], cl = second_date[-rows])
roc(y_h, as.numeric(second_date[-rows]))$auc




