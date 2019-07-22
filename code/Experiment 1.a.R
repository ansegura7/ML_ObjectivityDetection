###############################################################################################################
# Course: MINE-4205 - Knowledge Discovery from Social and Information Networks                                #
# Title: Implementation for objectivity detection                                                             #
# Authors: Segura Tinoco, German Andres and Cuevas Saavedra, Vladimir Enrique                                 #
# Date: July 10, 2018                                                                                         #
# Experiment: 1.a                                                                                             #
###############################################################################################################

# Delete variables from memory
rm(list = ls(all = TRUE))

#########################################
# 1. Loading libraries and source data  #
#########################################

# Loading R libraries
suppressWarnings(suppressMessages(library(e1071)))
suppressWarnings(suppressMessages(library(kknn)))
suppressWarnings(suppressMessages(library(MASS)))
suppressWarnings(suppressMessages(library(class)))
suppressWarnings(suppressMessages(library(rpart)))
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

# The Correlation circle is plotted - Only variables that have cos2 > 0.25 (25%)
plot(pca.res, axes=c(1, 2), choix="var", col.var="blue",new.plot=TRUE, select="cos2 0.25")

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
cat("# Objective: ", nObj, ", # Subjective: ", nSubj, ", % Objective Class: ", nObjClass)

#############################################
# 4. Training and Calibration of the model  #
#############################################

# Start the ML process
ptm <- proc.time()

# Get Target index
ixTypeVar = ncol(full_data)

# Cross-Validation iters
nCV <- 5

# K-folds number
nFolds <- 5

# Variables to store the detection of Label=subjective
detection.subj.svm <- rep(0, nCV)
detection.subj.knn <- rep(0, nCV)
detection.subj.bayes <- rep(0, nCV)
detection.subj.dtree <- rep(0, nCV)
detection.subj.forest <- rep(0, nCV)
detection.subj.adaboost <- rep(0, nCV)

# Variables to store the detection of Label=objective
detection.obj.svm <- rep(0, nCV)
detection.obj.knn <- rep(0, nCV)
detection.obj.bayes <- rep(0, nCV)
detection.obj.dtree <- rep(0, nCV)
detection.obj.forest <- rep(0, nCV)
detection.obj.adaboost <- rep(0, nCV)

# Variables to store the average global error
error.global.svm <- rep(0, nCV)
error.global.knn <- rep(0, nCV)
error.global.bayes <- rep(0, nCV)
error.global.dtree <- rep(0, nCV)
error.global.forest <- rep(0, nCV)
error.global.adaboost <- rep(0, nCV)

# Start cross-validation
for(i in 1:nCV) {
  cat("ix", i)
  
  # Groups are created (for this case 5)
  groups <- createFolds(1:nRows, k = nFolds)
  
  # Internal counters of the prediction of the subjective
  subj.svm <- 0
  subj.knn <- 0
  subj.bayes <- 0
  subj.dtree <- 0
  subj.forest <- 0
  subj.adaboost <- 0
  
  # Internal counters of the prediction of the objective
  obj.svm <- 0
  obj.knn <- 0
  obj.bayes <- 0
  obj.dtree <- 0
  obj.forest <- 0
  obj.adaboost <- 0
  
  # Internal error counters of the models
  error.svm <- 0
  error.knn <- 0
  error.bayes <- 0
  error.dtree <- 0
  error.forest <- 0
  error.adaboost <- 0
  
  # Loop to perform the cross-validation
  for(k in 1:nFolds) {
    
    # The learning and testing datasets are generated
    sample <- groups[[k]]
    tlearning <- full_data[-sample, ]
    ttesting <- full_data[sample, ]
    
    # 1. A model is generated with the Vector Support Machines method
    cat("- svm, ix:", i, ", k:", k)
    model <- svm(Label~., data = tlearning, kernel = "linear")
    prediction <- predict(model, ttesting)
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.svm <- subj.svm + MC[2, 2]
    obj.svm <- obj.svm + MC[1, 1]
    error.svm <- error.svm + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 2. A model is generated with the Nearest Neighbors method with K = 5
    cat("- knn, ix:", i, ", k:", k)
    model <- train.kknn(Label~.,data = tlearning, kmax = 5)
    prediction <- predict(model, ttesting[,-ixTypeVar])
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.knn <- subj.knn + MC[2, 2]
    obj.knn <- obj.knn + MC[1, 1]
    error.knn <- error.knn + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 3. A model is generated with the Naive Bayes method
    cat("- bayes, ix:", i, ", k:", k)
    model <- naiveBayes(Label~., data = tlearning)
    prediction <- predict(model, ttesting[,-ixTypeVar])
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.bayes <- subj.bayes + MC[2, 2]
    obj.bayes <- obj.bayes + MC[1, 1]
    error.bayes <- error.bayes + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 4. A model is generated with the Decision Trees method
    cat("- btree, ix:", i, ", k:", k)
    model <- rpart(Label~., data = tlearning)
    prediction <- predict(model, ttesting, type='class')
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.dtree <- subj.dtree + MC[2, 2]
    obj.dtree <- obj.dtree + MC[1, 1]
    error.dtree <- error.dtree + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 5. A model is generated with the Random Forest method with 300 Trees
    cat("- rforest, ix:", i, ", k:", k)
    model <- randomForest(Label~., data = tlearning, importance = TRUE, ntree = 300)
    prediction <- predict(model, ttesting[,-ixTypeVar])
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.forest <- subj.forest + MC[2, 2]
    obj.forest <- obj.forest + MC[1, 1]
    error.forest <- error.forest + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 6. A model is generated with the ADA Boosting method
    cat("- boost, ix:", i, ", k:", k)
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
  detection.subj.svm[i] <- subj.svm
  detection.subj.knn[i] <- subj.knn
  detection.subj.bayes[i] <- subj.bayes
  detection.subj.dtree[i] <- subj.dtree
  detection.subj.forest[i] <- subj.forest
  detection.subj.adaboost[i] <- subj.adaboost
  
  # The amount of objective detected by each method is saved
  detection.obj.svm[i] <- obj.svm
  detection.obj.knn[i] <- obj.knn
  detection.obj.bayes[i] <- obj.bayes
  detection.obj.dtree[i] <- obj.dtree
  detection.obj.forest[i] <- obj.forest
  detection.obj.adaboost[i] <- obj.adaboost
  
  # The average global error is saved for each model
  error.global.svm[i] <- error.svm / nFolds
  error.global.knn[i] <- error.knn / nFolds
  error.global.bayes[i] <- error.bayes / nFolds
  error.global.dtree[i] <- error.dtree / nFolds
  error.global.forest[i] <- error.forest / nFolds
  error.global.adaboost[i] <- error.adaboost / nFolds
}

###################################
# 5. Plotting the Models Results  #
###################################

# The limits for Plot 1 are calculated
yLim <- c(min(error.global.svm, error.global.knn, error.global.bayes, error.global.dtree, error.global.forest, error.global.adaboost) * 0.9,
          max(error.global.svm, error.global.knn, error.global.bayes, error.global.dtree, error.global.forest, error.global.adaboost) * 1.3)

# The curves are plotted with the average global error of each model
plot(error.global.svm, col = "magenta", type = "b", ylim = yLim, main = "Models error for subjectivity analysis",
     xlab = "Número de iteración", ylab = "Error Global", cex.axis = 0.5)
points(error.global.knn, col = "blue", type = "b")
points(error.global.bayes, col = "red", type = "b")
points(error.global.dtree, col = "lightblue3", type = "b")
points(error.global.forest, col = "olivedrab", type = "b")
points(error.global.adaboost, col = "orange3", type = "b")

# The legend is added
legend("topright", legend = c("SVM", "KNN", "Bayes", "Árbol", "Bosque", "AdaBoost"),
       col = c("magenta", "blue", "red", "lightblue3", "olivedrab", "orange3"),
       lty = 1, lwd = 2, ncol = 6, cex = 0.45)

# The limits for Plot 2 are setted
yLim <- c(nSubj * 0.5, nSubj * 1.05)

# The curves of the subjective detection are plotted
plot(detection.subj.svm, col = "magenta", type = "b", ylim = yLim, main = "Correct detection of subjective articles",
     xlab = "Número de iteración", ylab = "Cantidad de SI detectados", cex.axis = 0.5)
points(detection.subj.knn, col = "blue", type = "b")
points(detection.subj.bayes, col = "red", type = "b")
points(detection.subj.dtree, col = "lightblue3", type = "b")
points(detection.subj.forest, col = "olivedrab", type = "b")
points(detection.subj.adaboost, col = "orange3", type = "b")

# The legend is added
legend("topright", legend = c("SVM", "KNN", "Bayes", "Árbol", "Bosque", "AdaBoost"),
       col = c("magenta", "blue", "red", "lightblue3", "olivedrab", "orange3"),
       lty = 1, lwd = 2, ncol = 6, cex = 0.45)

# The limits for Plot 3 are setted
yLim <- c(nObj * 0.5, nObj * 1.05)

# The curves of the objective detection are plotted
plot(detection.obj.svm, col = "magenta", type = "b", ylim = yLim, main = "Correct detection of objective articles",
     xlab = "Número de iteración", ylab = "Cantidad de NO detectados", cex.axis = 0.5)
points(detection.obj.knn, col = "blue", type = "b")
points(detection.obj.bayes, col = "red", type = "b")
points(detection.obj.dtree, col = "lightblue3", type = "b")
points(detection.obj.forest, col = "olivedrab", type = "b")
points(detection.obj.adaboost, col = "orange3", type = "b")

# The legend is added
legend("topright", legend = c("SVM", "KNN", "Bayes", "Árbol", "Bosque", "AdaBoost"),
       col = c("magenta", "blue", "red", "lightblue3", "olivedrab", "orange3"),
       lty = 1, lwd = 2, ncol = 6, cex = 0.45)

# The average Errors of the models are shown
print(mean(error.global.svm))
print(mean(error.global.knn))
print(mean(error.global.bayes))
print(mean(error.global.dtree))
print(mean(error.global.forest))
print(mean(error.global.adaboost))

# The subjective detected by each models are shown
print(mean(detection.subj.svm))
print(mean(detection.subj.knn))
print(mean(detection.subj.bayes))
print(mean(detection.subj.dtree))
print(mean(detection.subj.forest))
print(mean(detection.subj.adaboost))

# The objective detected by each models are shown
print(mean(detection.obj.svm))
print(mean(detection.obj.knn))
print(mean(detection.obj.bayes))
print(mean(detection.obj.dtree))
print(mean(detection.obj.forest))
print(mean(detection.obj.adaboost))

# End of the ML process
ptm <- (proc.time() - ptm)
cat("Time: ", ptm)

#####################
# 6. End of Script  #
#####################