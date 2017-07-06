#Chapter 8 Code Examples

#Clear the workspace 
rm(list = ls())

#Loading Data 
data("iris")

#Simple ANOVA
#Toy Example Using Iris Data as Y
y <- iris[, 1]
x <- seq(1, length(y), 1)

#Plotting Data
plot(y)

#Plotting Residuals
par(mfrow = c(2,2))
plot(glm(y~x))
dev.off()

simpleAOV <- aov(y ~ x)
summary(simpleAOV)

#Mixed Design Anova
x1 <- iris[,2]
x2 <- iris[,3]

mixedAOV <- aov(y ~ x1*x2)
summary(mixedAOV)


#Residuals: Mixed Design Anova
par(mfrow = c(2,2))
plot(glm(y ~ x1*x2))
dev.off()

#Multiple ANOVA 
multAOV <- manova(cbind(x1,x2) ~ y)
summary(multAOV)