#Introduction to Deep Learning
#Image Download and Preprocessing
#Taweh Beysolow II 

#Clear the workspace
rm(list = ls())

#In this example, we will walk the readers through building a convolutional neural network
#and then training it. We will begin by loading a subset of the the caltech image library 

#Loading required packages
require(mxnet)
require(EBImage)
require(jpeg)
require(pROC)

#Downloading the strings of the image files in each directory
guitar_photos <- list.files("/Users/tawehbeysolow/Downloads/101_ObjectCategories/electric_guitar")
laptop_photos <- list.files("/Users/tawehbeysolow/Downloads/101_ObjectCategories/laptop")

##################################################################################################
#Preprocessing
##################################################################################################
#Downloading the image data 
img_data <- data.frame()

#Turning Photos into Bitmaps
#Guitar Bitmaps
for (i in 1:length(guitar_photos)){
  img <- readJPEG(paste("/Users/tawehbeysolow/Downloads/101_ObjectCategories/electric_guitar/", guitar_photos[i], sep = ""))
  
  #Reshape to 64x64 pixel size and grayscale image
  img <- Image(img, dim = c(64, 64), color = "grayscale")
  
  #Resizing Image to 28x28 Pixel Size
  img <- resize(img, w = 28, h = 28)
  img <- img@.Data
  
  #Transforming to vector
  img <- as.vector(t(img))
  
  #Adding Label 
  label <- 1
  
  img <- c(label, img)

  #Appending to List
 img_data <- rbind(img_data, img)
 
}

#Laptop Bitmaps
for (i in 1:length(laptop_photos)){
  img <- readJPEG(paste("/Users/tawehbeysolow/Downloads/101_ObjectCategories/laptop/", laptop_photos[i], sep = ""))
  
  #Reshape to 64x64 pixel size and grayscale image
  img <- Image(img, dim = c(64, 64), color = "grayscale")
  
  #Resizing Image to 28x28 Pixel Size
  img <- resize(img, w = 28, h = 28)
  img <- img@.Data
  
  #Transforming to vector
  img <- as.vector(t(img))
  
  #Adding Label 
  label <- 2
  
  img <- c(label, img)
  
  #Appending to List
  img_data <- rbind(img_data, img)
  
}

#Transforming data into matrix for input into CNN 
training_set <- data.matrix(img_data)


#Cross Validating Results 
rows <- sample(1:nrow(training_set), nrow(training_set)*.75)

#Training Set
x_train <- t(training_set[rows, -1])
y_train <- training_set[rows, 1]

dim(x_train) <- c(28,28, 1, ncol(x_train))

#Test Set
x_test <- t(training_set[-rows, -1])
y_test <- training_set[-rows, 1]

dim(x_test) <- c(28,28, 1, ncol(x_test))

##################################################################################################
#Building Convolutional Neural Network 
#We will use a LeNet Architecture for this example. Readers may feel free to experiment by 
#using alternative arhcitectures 

data <- mx.symbol.Variable('data')

#Layer 1
convolution_l1 <- mx.symbol.Convolution(data = data, kernel = c(5,5), num_filter = 20)
tanh_l1 <- mx.symbol.Activation(data = convolution_l1, act_type = "tanh")
pooling_l1 <- mx.symbol.Pooling(data = tanh_l1, pool_type = "max", kernel = c(2,2), stride = c(2,2))

#Layer 2
convolution_l2 <- mx.symbol.Convolution(data = pooling_l1, kernel = c(5,5), num_filter = 20)
tanh_l2 <- mx.symbol.Activation(data = convolution_l2, act_type = "tanh")
pooling_l2 <- mx.symbol.Pooling(data = tanh_l2, pool_type = "max", kernel = c(2,2), stride = c(2,2))

#Fully Connected 1
fl <- mx.symbol.Flatten(data = pooling_l2)
full_conn1 <- mx.symbol.FullyConnected(data = fl, num_hidden = 500)
tanh_l3 <- mx.symbol.Activation(data = full_conn1, act_type = "tanh")

#Fully Connected 2
full_conn2 <- mx.symbol.FullyConnected(data = tanh_l3, num_hidden = 40)

#Softmax Classification Layer 
CNN <- mx.symbol.SoftmaxOutput(data = full_conn2)

##################################################################################################
#Model Training and Parameter Tuning
mx.set.seed(2017)
#Learning Rate Parameter
AUC <- c()
learn_rate <- c(0.01, 0.02, 0.03, 0.04)
CPU <- mx.cpu()

for (i in 1:length(learn_rate)){
  
  cnn_model <- mx.model.FeedForward.create(CNN, X = x_train, y = y_train, ctx = CPU,
                                           num.round = 50, array.batch.size = 40,
                                           learning.rate = learn_rate[i],
                                           momentum = 0.9, eval.metric = mx.metric.accuracy,
                                           epoch.end.callback = mx.callback.log.train.metric(100), optimizer = "sgd")
  #Calculating Training Accuracy
  y_h <- predict(cnn_model, x_train)
  Labels <- max.col(t(y_h)) - 1
  AUC <- append(AUC, roc(as.factor(y_train), as.numeric(Labels))$auc[1])

}

#Plotting AUC
plot(learn_rate, AUC, main = "AUC for CNN \n Training Learning Rate Parameter", xlab = "learning rate", 
     ylab = "AUC Score", type = "l", col = "cadetblue")

#Momentum Parameter
mx.set.seed(2017)
AUC1 <- c()
mom <- c(0.5, 0.9, 1.5)
CPU <- mx.cpu()

for (i in 1:length(mom)){
  
  cnn_model <- mx.model.FeedForward.create(CNN, X = x_train, y = y_train, ctx = CPU,
                                           num.round = 50, array.batch.size = 40,
                                           learning.rate = 0.04,
                                           momentum = mom[i], eval.metric = mx.metric.accuracy,
                                           epoch.end.callback = mx.callback.log.train.metric(100), optimizer = "sgd")
  #Calculating Training Accuracy
  y_h <- predict(cnn_model, x_train)
  Labels <- max.col(t(y_h)) - 1
  AUC1 <- append(AUC1, roc(as.factor(y_train), as.numeric(Labels))$auc[1])
  
}

#Plotting AUC
plot(mom, AUC1, main = "AUC for CNN \n Training Momentum Parameter", xlab = "momentum", 
     ylab = "AUC Score", type = "l", col = "cadetblue")


##################################################################################################
#Fitted Model Training
cnn_model <- mx.model.FeedForward.create(CNN, X = x_train, y = y_train, ctx = CPU,
                                         num.round = 150, array.batch.size = 40,
                                         learning.rate = 0.04, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                         optimizer = "sgd")
#Calculating Training Set Accuracy
y_h <- predict(cnn_model, x_train)
Labels <- max.col(t(y_h)) - 1
roc(as.factor(y_train), as.numeric(Labels))
curve <- roc(as.factor(y_train), as.numeric(Labels))

#Plotting Results
plot(curve, main = "ROC Curve for Convolutional Neural Network \n Train Set")

#Calculating Test Set Accuracy
y_h <- predict(cnn_model, x_test)
Labels <- max.col(t(y_h)) - 1
roc(as.factor(y_test), as.numeric(Labels))
curve1 <- roc(as.factor(y_test), as.numeric(Labels))

#Plotting Results
plot(curve1, main = "ROC Curve for Convolutional Neural Network \n Test Set")
