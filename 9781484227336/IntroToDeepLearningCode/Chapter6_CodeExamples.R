#Chapter 6 Code Examples
#Recurrent Neural Networks for Sequence Prediction 

#Clear the workspace 
rm(list = ls())

#Load the necessary packages 
require(rnn)
require(Metrics)

#Function to be used later
#Creating Training and Test Data Set
dataset <- function(data){
  x <- y <- c()
  for (i in 1:(nrow(data)-2)){
    x <- append(x, data[i, 2])
    y <- append(y, data[i+1, 2])
  }
  #Creating New DataFrame
  output <- cbind(x,y)
  return(output[1:nrow(output)-1,])
}

#Monthly Milk Production: Pounds Per Cow 
data <- read.table("/Users/tawehbeysolow/Downloads/monthly-milk-production-pounds-p.csv", header = TRUE, sep = ",")

#Plotting Sequence
plot(data[,2], main = "Monthly Milk Production in Pounds", xlab = "Month", ylab = "Pounds", 
     lwd = 1.5, col = "cadetblue", type = "l")

#Ploting Histogram
hist(data[,2], main = "Histogram of Monthly Milk Production in Pounds", xlab = "Pounds", col = "red")

#Creating Test and Training Sets
newData <- dataset(data = data)
rows <- sample(1:120, 120)
trainingData <- scale(newData[rows, ])
testData <- scale(newData[-rows, ])

#Max-Min Scaling
x <- trainingData[,1]
y <- trainingData[,2]
train_x <- (x - min(x))/(max(x)-min(x))
train_y <- (y - min(y))/(max(y)-min(y))

#RNN Model 
RNN <- trainr(Y = as.matrix(train_x),
               X = as.matrix(train_y),
              learningrate = 0.04, 
              momentum = 0.1,
              network_type = "rnn",
              numepochs = 700,
              hidden_dim = 3)

y_h <- predictr(RNN, as.matrix(train_x))

#Comparing Plots of Predicted Curve vs Actual Curve: Training Data 
plot(train_y, col = "blue", type = "l", main = "Actual vs Predicted Curve: Training Data", lwd = 2)
lines(y_h, type = "l", col = "red", lwd = 2)
cat("Train MSE: ", mse(y_h, train_y))

#Test Data 
testData <- scale(newData[-rows, ])
x <- testData[,1]
y <- testData[,2]
test_x <- (x - min(x))/(max(x)-min(x))
test_y <- (y - min(y))/(max(y)-min(y))
y_h2 <- predictr(RNN, as.matrix(x))

#Comparing Plots of Predicted Curve vs Actual Curve: Training Data 
plot(test_y, col = "blue", type = "l", main = "Actual vs Predicted Curve: Test Data", lwd = 2)
lines(y_h2, type = "l", col = "red", lwd = 2)
cat("Test MSE: ", mse(y_h2, test_y))

