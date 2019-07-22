###############################################################################################################
# Course: MINE-4205 - Knowledge Discovery from Social and Information Networks                                #
# Title: Implementation for objectivity detection                                                             #
# Authors: Segura Tinoco, German Andres and Cuevas Saavedra, Vladimir Enrique                                 #
# Date: July 10, 2018                                                                                         #
# Experiment: 1.b                                                                                             #
###############################################################################################################

# Delete variables from memory
rm(list = ls(all = TRUE))

#########################################
# 1. Loading libraries and source data  #
#########################################

# Loading R libraries
suppressWarnings(suppressMessages(library(e1071)))
suppressWarnings(suppressMessages(library(MASS)))
suppressWarnings(suppressMessages(library(class)))
suppressWarnings(suppressMessages(library(randomForest)))
suppressWarnings(suppressMessages(library(ada)))

# Loading K-folds library
suppressMessages(suppressMessages(library(caret)))

# Loading FactoMineR library
suppressMessages(library(FactoMineR))

# Set work directory
setwd("C:/data/")

# Loading data
full_data <- read.csv("features.csv", header = TRUE, sep = ',', dec = '.')

#################################
# 2. Descriptive Data Analysis  #
#################################

# Show dataframe dim(rows, cols)
dim(full_data)

# Summary of dataset
head(full_data, n = 10)

# Boxplots are created to identify the outliers for each variable
boxplot(x = full_data, range = 2, border = c("blue", "green", "black", "orange"), las=2, cex.axis=0.9, cex.names=0.9)

# Apply PCA
pca.res <- PCA(full_data[-59], scale.unit = TRUE, graph = FALSE)
pca.res$eig

##############################################
# 3. The Base Reference Error is calculated  #
##############################################

# Get rows number
nRows <- nrow(full_data)

# Calculate the amount of Objective and Subjective labels
nSubj <- sum(full_data$Label == "subjective")
nObj <- nRows - nSubj

# The presence percentage of the Objective class is calculated
nObjClass <- nObj * 100 / nRows
cat("# Objective: ", nObj, ", # Subjective: ", nSubj, ", % Objective Class: ", nObjClass, "\n")

#############################################
# 4. Training and Calibration of the model  #
#############################################

# Start the ML process
ptm <- proc.time()

# Get Target index
ixTypeVar = ncol(full_data)

# Cross-Validation iters
nCV <- 10

# K-folds number
nFolds <- 10

# Variables to store the detection of Label=subjective
detection.subj.bayes <- rep(0, nCV)
detection.subj.svm.lin <- rep(0, nCV)
detection.subj.svm.rad <- rep(0, nCV)
detection.subj.forest.200 <- rep(0, nCV)
detection.subj.forest.300 <- rep(0, nCV)
detection.subj.adaboost <- rep(0, nCV)

# Variables to store the detection of Label=objective
detection.obj.bayes <- rep(0, nCV)
detection.obj.svm.lin <- rep(0, nCV)
detection.obj.svm.rad <- rep(0, nCV)
detection.obj.forest.200 <- rep(0, nCV)
detection.obj.forest.300 <- rep(0, nCV)
detection.obj.adaboost <- rep(0, nCV)

# Variables to store the average global error
error.global.bayes <- rep(0, nCV)
error.global.svm.lin <- rep(0, nCV)
error.global.svm.rad <- rep(0, nCV)
error.global.forest.200 <- rep(0, nCV)
error.global.forest.300 <- rep(0, nCV)
error.global.adaboost <- rep(0, nCV)

# Start cross-validation
for(i in 1:nCV) {
  cat("ix", i)
  
  # Groups are created (for this case 10)
  groups <- createFolds(1:nRows, k = nFolds)
  
  # Internal counters of the prediction of the subjective
  subj.bayes <- 0
  subj.svm.lin <- 0
  subj.svm.rad <- 0
  subj.forest.200 <- 0
  subj.forest.300 <- 0
  subj.adaboost <- 0
  
  # Internal counters of the prediction of the objective
  obj.bayes <- 0
  obj.svm.lin <- 0
  obj.svm.rad <- 0
  obj.forest.200 <- 0
  obj.forest.300 <- 0
  obj.adaboost <- 0
  
  # Internal error counters of the models
  error.bayes <- 0
  error.svm.lin <- 0
  error.svm.rad <- 0
  error.forest.200 <- 0
  error.forest.300 <- 0
  error.adaboost <- 0
  
  # Loop to perform the cross-validation
  for(k in 1:nFolds) {
    
    # The learning and testing datasets are generated
    sample <- groups[[k]]
    tlearning <- full_data[-sample, ]
    ttesting <- full_data[sample, ]
    
    # 1. A model is generated with the Naive Bayes method
    cat("- bayes, ix:", i, ", k:", k, "\n")
    model <- naiveBayes(Label~., data = tlearning)
    prediction <- predict(model, ttesting[,-ixTypeVar])
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.bayes <- subj.bayes + MC[2, 2]
    obj.bayes <- obj.bayes + MC[1, 1]
    error.bayes <- error.bayes + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 2. A model is generated with the Vector Support Machines method
    cat("- svm linear, ix:", i, ", k:", k, "\n")
    model <- svm(Label~., data = tlearning, kernel = "linear")
    prediction <- predict(model, ttesting)
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.svm.lin <- subj.svm.lin + MC[2, 2]
    obj.svm.lin <- obj.svm.lin + MC[1, 1]
    error.svm.lin <- error.svm.lin + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 3. A model is generated with the Nearest Neighbors method with K = 5
    cat("- svm radial, ix:", i, ", k:", k, "\n")
    model <- svm(Label~., data = tlearning, kernel="polynomial", gama=0.25, cost=1)
    prediction <- predict(model, ttesting)
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.svm.rad <- subj.svm.rad + MC[2, 2]
    obj.svm.rad <- obj.svm.rad + MC[1, 1]
    error.svm.rad <- error.svm.rad + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 4. A model is generated with the Decision Trees method
    cat("- rforest 200, ix:", i, ", k:", k, "\n")
    model <- randomForest(Label~., data = tlearning, importance = TRUE, ntree = 200)
    prediction <- predict(model, ttesting[,-ixTypeVar])
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.forest.200 <- subj.forest.200 + MC[2, 2]
    obj.forest.200 <- obj.forest.200 + MC[1, 1]
    error.forest.200 <- error.forest.200 + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 5. A model is generated with the Random Forest method with 300 Trees
    cat("- rforest 300, ix:", i, ", k:", k, "\n")
    model <- randomForest(Label~., data = tlearning, importance = TRUE, ntree = 300)
    prediction <- predict(model, ttesting[,-ixTypeVar])
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.forest.300 <- subj.forest.300 + MC[2, 2]
    obj.forest.300 <- obj.forest.300 + MC[1, 1]
    error.forest.300 <- error.forest.300 + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 6. A model is generated with the ADA Boosting method
    cat("- ada boost, ix:", i, ", k:", k, "\n")
    model <- ada(Label~., data = tlearning, iter = 50, nu = 1, type="real")
    prediction <- predict(model, ttesting[,-ixTypeVar])
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.adaboost <- subj.adaboost + MC[2, 2]
    obj.adaboost <- obj.adaboost + MC[1, 1]
    error.adaboost <- error.adaboost + (1 - (sum(diag(MC)) / sum(MC))) * 100
  }
  
  # The amount of subjective detected by each method is saved
  detection.subj.bayes[i] <- subj.bayes
  detection.subj.svm.lin[i] <- subj.svm.lin
  detection.subj.svm.rad[i] <- subj.svm.rad
  detection.subj.forest.200[i] <- subj.forest.200
  detection.subj.forest.300[i] <- subj.forest.300
  detection.subj.adaboost[i] <- subj.adaboost
  
  # The amount of objective detected by each method is saved
  detection.obj.bayes[i] <- obj.bayes
  detection.obj.svm.lin[i] <- obj.svm.lin
  detection.obj.svm.rad[i] <- obj.svm.rad
  detection.obj.forest.200[i] <- obj.forest.200
  detection.obj.forest.300[i] <- obj.forest.300
  detection.obj.adaboost[i] <- obj.adaboost
  
  # The average global error is saved for each model
  error.global.bayes[i] <- error.bayes / nFolds
  error.global.svm.lin[i] <- error.svm.lin / nFolds
  error.global.svm.rad[i] <- error.svm.rad / nFolds
  error.global.forest.200[i] <- error.forest.200 / nFolds
  error.global.forest.300[i] <- error.forest.300 / nFolds
  error.global.adaboost[i] <- error.adaboost / nFolds
}

###################################
# 5. Plotting the Models Results  #
###################################

# The limits for Plot 1 are calculated
yLim <- c(min(error.global.bayes, error.global.svm.lin, error.global.svm.rad, error.global.forest.200, error.global.forest.300, error.global.adaboost) * 0.9,
          max(error.global.bayes, error.global.svm.lin, error.global.svm.rad, error.global.forest.200, error.global.forest.300, error.global.adaboost) * 1.3)

# The curves are plotted with the average global error of each model
plot(error.global.bayes, col = "magenta", type = "b", ylim = yLim, main = "Models error for subjectivity analysis",
     xlab = "Número de iteración", ylab = "Error Global", cex.axis = 0.5)
points(error.global.svm.lin, col = "red", type = "b")
points(error.global.svm.rad, col = "blue", type = "b")
points(error.global.forest.200, col = "lightblue3", type = "b")
points(error.global.forest.300, col = "olivedrab", type = "b")
points(error.global.adaboost, col = "orange3", type = "b")

# The legend is added
legend("topright", legend = c("Bayes", "SVM Lin", "SVM Rad", "Forest 200", "Forest 300", "AdaBoost"),
       col = c("magenta", "blue", "red", "lightblue3", "olivedrab", "orange3"),
       lty = 1, lwd = 2, ncol = 6, cex = 0.45)

# The limits for Plot 2 are setted
yLim <- c(nSubj * 0.5, nSubj * 1.05)

# The curves of the subjective detection are plotted
plot(detection.subj.bayes, col = "magenta", type = "b", ylim = yLim, main = "Correct detection of subjective articles",
     xlab = "Número de iteración", ylab = "Cantidad de SI detectados", cex.axis = 0.5)
points(detection.subj.svm.lin, col = "red", type = "b")
points(detection.subj.svm.rad, col = "blue", type = "b")
points(detection.subj.forest.200, col = "lightblue3", type = "b")
points(detection.subj.forest.300, col = "olivedrab", type = "b")
points(detection.subj.adaboost, col = "orange3", type = "b")

# The legend is added
legend("topright", legend = c("Bayes", "SVM Lin", "SVM Rad", "Forest 200", "Forest 300", "AdaBoost"),
       col = c("magenta", "blue", "red", "lightblue3", "olivedrab", "orange3"),
       lty = 1, lwd = 2, ncol = 6, cex = 0.45)

# The limits for Plot 3 are setted
yLim <- c(nObj * 0.5, nObj * 1.05)

# The curves of the objective detection are plotted
plot(detection.obj.bayes, col = "magenta", type = "b", ylim = yLim, main = "Correct detection of objective articles",
     xlab = "Número de iteración", ylab = "Cantidad de NO detectados", cex.axis = 0.5)
points(detection.obj.svm.lin, col = "red", type = "b")
points(detection.obj.svm.rad, col = "blue", type = "b")
points(detection.obj.forest.200, col = "lightblue3", type = "b")
points(detection.obj.forest.300, col = "olivedrab", type = "b")
points(detection.obj.adaboost, col = "orange3", type = "b")

# The legend is added
legend("topright", legend = c("Bayes", "SVM Lin", "SVM Rad", "Forest 200", "Forest 300", "AdaBoost"),
       col = c("magenta", "blue", "red", "lightblue3", "olivedrab", "orange3"),
       lty = 1, lwd = 2, ncol = 6, cex = 0.45)

# The average Errors of the models are shown
print(mean(error.global.bayes))
print(mean(error.global.svm.lin))
print(mean(error.global.svm.rad))
print(mean(error.global.forest.200))
print(mean(error.global.forest.300))
print(mean(error.global.adaboost))

# The subjective detected by each models are shown
print(mean(detection.subj.bayes))
print(mean(detection.subj.svm.lin))
print(mean(detection.subj.svm.rad))
print(mean(detection.subj.forest.200))
print(mean(detection.subj.forest.300))
print(mean(detection.subj.adaboost))

# The objective detected by each models are shown
print(mean(detection.obj.bayes))
print(mean(detection.obj.svm.lin))
print(mean(detection.obj.svm.rad))
print(mean(detection.obj.forest.200))
print(mean(detection.obj.forest.300))
print(mean(detection.obj.adaboost))

# End of the ML process
ptm <- (proc.time() - ptm)
cat("Time: ", ptm)

#####################
# 6. End of Script  #
#####################