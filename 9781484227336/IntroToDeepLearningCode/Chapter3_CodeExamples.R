#An Introduction to Deep Learning
#Chapter 3 
#Clear the workspace 
rm(list = ls())

#Upload Necessary Packages
require(MASS)

#Modifying Data From Iris Data Set 
data(iris)
Y <-  matrix(iris[,1])
X <-  matrix(seq(0,149, 1))

olsExample <- function(y = Y, x = X){
  y_h <- lm(y ~ x)
  y_hf <- y_h$fitted.values
  error <- sum((y_hf - y)^2)
  coefs <- y_h$coefficients
  output <- list("Cost" = error, "Coefficients" = coefs)
  return(output)
}

#Gradient Descent Without Adaptive Step
gradientDescent <- function(y = Y, x = X, alpha = .0001, epsilon = .001, maxiter = 300000){
  #Intializing Parameters
  theta0 <-  0
  theta1 <-  0
  cost <- sum(((theta0 + theta1*x) - y)^2)
  converged <- FALSE
  iterations <- 1
  
  #Gradient Descent Algorithm 
  while (converged == FALSE){
    gradient0 <- as.numeric((1/length(y))*sum((theta0 + theta1*x) - y))
    gradient1 <- as.numeric((1/length(y))*sum((((theta0 + theta1*x) - y)*x)))
    
    t0 <- as.numeric(theta0 - (alpha*gradient0))
    t1 <- as.numeric(theta1 - (alpha*gradient1))
    
    theta0 <- t0
    theta1 <- t1
    
    error <- as.numeric(sum(((theta0 + theta1*x) - y)^2))
    
    if (as.numeric(abs(cost - error)) <= epsilon){
      converged <- TRUE
    }
      cost <- error
      iterations <- iterations + 1
    if (iterations == maxiter){
      converged <- TRUE
    }
  }
  output <- list("theta0" = theta0, "theta1" = theta1, "Cost" = cost, "Iterations" = iterations)
  return(output)
}

#Gradient Descent With Adaptive Step
adaptiveGradient <- function(y = Y, x = X, alpha = .0001, epsilon = .000001, maxiter = 300000){
  #Intializing Parameters
  theta0 <-  0
  theta1 <-  0
  cost <- sum(((theta0 + theta1*x) - y)^2)
  converged <- FALSE
  iterations <- 1

  #Gradient Descent Algorithm 
  while (converged == FALSE){
    gradient0 <- as.numeric((1/length(y))*sum((theta0 + theta1*x) - y))
    gradient1 <- as.numeric((1/length(y))*sum((((theta0 + theta1*x) - y)*x)))
    t0 <- as.numeric(theta0 - (alpha*gradient0))
    t1 <- as.numeric(theta1 - (alpha*gradient1))
    delta_0 <- t0 - theta0
    
    if (delta_0 < theta0){
      alpha <- alpha*1.10
    } else {
      alpha <- alpha*.50
    }
    
    theta0 <- t0
    theta1 <- t1
    error <- as.numeric(sum(((theta0 + theta1*x) - y)^2))
    
    if (as.numeric(abs(cost - error)) <= epsilon){
      converged <- TRUE
    }
    cost <- error
    iterations <- iterations + 1
    if (iterations == maxiter){
      converged <- TRUE
    }
  }
  output <- list("theta0" = theta0, "theta1" = theta1, "Cost" = cost, "Iterations" = iterations, "Learning.Rate" = alpha)
  return(output)
}

#Ridge Regression Function
ridgeRegression <- function(y = Y, x = cbind(X, X), lambda = 1){
  I  <- diag(ncol(x))
  gamma  <- lambda*I
  beta_h  <- (ginv(t(x)%*%x + t(gamma)%*%gamma)) %*% (t(x) %*% y)
  beta_0 <- mean(y) - mean(x)*beta_h
  x <- data.frame(x)
  y_h <- beta_0 + (x*beta_h)
  RSS <- sum((y - y_h)^2)
  output <- list("Cost" = RSS, "Theta0" = beta_0, "Theta1" = beta_h)
  return(output)
}

olsExample()
gradientDescent()
adaptiveGradient()
ridgeRegression()


#Logistic Regression 
#Clear the workspace
rm(list = ls())

#Find Data Set on Github under AnIntroductionToDeepLearning Repository
#Upload Necessary Packages 
require(ggplot2)
require(lattice)
require(nnet)
require(pROC)
require(ROCR)

#Upload the necessary data 
data  <- read.csv("/Users/tawehbeysolow/Desktop/projectportfolio/SpeedDating.csv", header = TRUE, stringsAsFactors = TRUE)

#Creating Repsponse Variable
second_date  <- ifelse(data[,1] + data[,2] == 2, 1, 0)
data  <- cbind(second_date, data)

#Transforming Charcter Vectors into Numerical vectors for feature selection
data$RaceM <- as.factor(data$RaceM)
data$RaceF <- as.factor(data$RaceF)
data$RaceM <- as.numeric(data$RaceM)
data$RaceF <- as.numeric(data$RaceF)

#Removing NA Values 
data <- data[complete.cases(data), ]

#Performing Variable Selection 
pca_data <- prcomp(data, scale = TRUE)
stdev_data <- pca_data$sdev
print(stdev_data)
n_col <- which(stdev_data >= 1.0)
data <- data[, n_col]

#Correlation Matrix
print(cor(data))

#Logistic Regression Model
lr1  <- glm(data[,1] ~ data[,2] + data[,3] + data[,4] + data[,5] + data[,6] + data[,7], 
            family = binomial(link = "logit"), data = data)
summary(lr1)

#Plotting Residuals 
par(mfrow = c(2,2))
plot(lr1)

#Building Random Threshold 
y_h <- ifelse(lr1$fitted.values >= .40, 1, 0)

#Confusion Matrix 
confusion_matrix <- table(y_h, data[,1])
print(confusion_matrix)

#Construct ROC Curve
roc(response = data[,1], predictor = y_h, plot=TRUE, las=TRUE,   legacy.axes=TRUE, lwd=5,
    main="ROC for Speed Dating Analysis", cex.main=1.6, cex.axis=1.3, cex.lab=1.3)


#K_Means Clustering 
#Upload data 
data  <- read.table("http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/nci.data", sep ="", header = FALSE)
data <- t(data)
k_means  <- c()
k  <-  seq(2, 10, 1)
for (i in k){
  k_means[i]  <- kmeans(data, i, iter.max = 100, nstart = i)$tot.withinss
}
clus <- kmeans(data, 10)$cluster
summ  <- table(clus)
#Removing NA Values
k_means  <- k_means[!is.na(k_means)]
#Plotting Sum of Squares over K 
plot(k, k_means,  main ="Sum of Squares Over K-Clusters", xlab = "K Clusters", ylab= "Sum of Squares",
     type = "b", col = "red")


#Support Vector Classificaiton 
require(LiblineaR)
require(e1071)

#Upload the data 
data  <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                    sep = ",", header = FALSE)
names(data)  <-  c("sepal.length", "sepal.width", "petal.length", "petal.width",
                   "iris.type")

x <- data[,1:2]
y <- factor(data[,5])
y <- as.numeric(y)
rows <- sample(1:nrow(data), 100, replace=FALSE)
x_train <- x[rows,]
y_train <- y[rows]
s  <- scale(x_train, center=TRUE, scale=TRUE)

#SVM Classification
output  <- LiblineaR(data=s, target=y_train, type = 3, cost = heuristicC(s))
#Predicted Y Values 
y_h <- predict(output, s, decisionValues = TRUE)$predictions

#Confusion Matrix 
confusion_matrix <- table(y_h, y_train)
print(confusion_matrix)


#Expectation-Maximization Algorithm for Clustering 
require(MASS)
require(mclust)
y_h <- Mclust(x_train, G = 3)$classification
print(table(y_h, y_train))
plot(Mclust(x_train, G = 3), what = c("classification"), dimens=c(1,3)) 
confusion_matrix <- table(y_h, y_train)
print(confusion_matrix)

#Classification Tree
require(rpart)
#Upload the necessary data 
data  <- read.csv("/Users/tawehbeysolow/Desktop/projectportfolio/SpeedDating.csv", header = TRUE, stringsAsFactors = TRUE)

#Creating Repsponse Variable
second_date  <- ifelse(data[,1] + data[,2] == 2, 1, 0)
data  <- cbind(data, second_date)

#Transforming Charcter Vectors into Numerical vectors for feature selection
data$RaceM <- as.factor(data$RaceM)
data$RaceF <- as.factor(data$RaceF)
data$RaceM <- as.numeric(data$RaceM)
data$RaceF <- as.numeric(data$RaceF)

#Removing NA Values 
data <- data[complete.cases(data), ]

#Performing Variable Selection 
pca_data <- prcomp(data, scale = TRUE)
stdev_data <- summary(pca_data)$importance[2,]
print(stdev_data)
n_col <- which(stdev_data >= .04)
data <- data[, n_col]

#Cross Validating Data 
rows <- sample(1:150, 150, replace = FALSE)
x_train <- data[rows, 2:ncol(data)]
y_train <- data[rows, 1]

#Correlation Matrix
classification_tree <- rpart(y_train ~ x_train[,1] + x_train[,2] + x_train[,3] + x_train[,4]
                             +x_train[,5] + x_train[,6], method = "class")
pruned_tree <- prune(classification_tree, cp = .01)

#Data Plot
plot(pruned_tree, uniform = TRUE, branch  = .7, margin = .1, cex = .08)
text(pruned_tree, all = TRUE, use.n = TRUE)

#Outputting Predictions
y_h <- predict(classification_tree, x_train, type = "class")
confusion_matrix <- table(y_h, y_train)
print(confusion_matrix)

#Bayesian Classifier 
require(e1071)
#Upload the necessary data 
data  <- read.csv("/Users/tawehbeysolow/Desktop/projectportfolio/SpeedDating.csv", header = TRUE, stringsAsFactors = TRUE)

#Creating Repsponse Variable
second_date  <- ifelse(data[,1] + data[,2] == 2, 1, 0)
data  <- cbind(data, second_date)

#Transforming Charcter Vectors into Numerical vectors for feature selection
data$RaceM <- as.factor(data$RaceM)
data$RaceF <- as.factor(data$RaceF)
data$RaceM <- as.numeric(data$RaceM)
data$RaceF <- as.numeric(data$RaceF)

#Removing NA Values 
data <- data[complete.cases(data), ]

#Performing Variable Selection 
pca_data <- prcomp(data, scale = TRUE)
stdev_data <- summary(pca_data)$importance[2,]
print(stdev_data)
n_col <- which(stdev_data >= .04)
data <- data[, n_col]

#Cross Validating Data 
rows <- sample(1:150, 150, replace = FALSE)
x_train <- data[rows, 2:ncol(data)]
for (i in 1:ncol(x_train)){
  x_train[,i] <- as.factor(x_train[,i])
}
y_train <- as.factor(data[rows, 1])

#Fitting Model
bayes_classifier <- naiveBayes(y = y_train, x = x_train , data = x_train)
y_h <- predict(bayes_classifier, x_train, type = c("class"))

#Evaluating Model 
confusion_matrix <- table(y_h, y_train)
print(confusion_matrix)

