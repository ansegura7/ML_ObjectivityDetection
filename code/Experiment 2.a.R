###############################################################################################################
# Course: MINE-4205 - Knowledge Discovery from Social and Information Networks                                #
# Title: Implementation for objectivity detection                                                             #
# Authors: Segura Tinoco, German Andres and Cuevas Saavedra, Vladimir Enrique                                 #
# Date: July 11, 2018                                                                                         #
# Experiment: 2.a                                                                                             #
###############################################################################################################

# Delete variables from memory
rm(list = ls(all = TRUE))

#########################################
# 1. Loading libraries and source data  #
#########################################

# Loading R libraries
suppressWarnings(suppressMessages(library(e1071)))

# Loading K-folds library
suppressMessages(suppressMessages(library(caret)))

# Loading FactoMineR library
suppressMessages(library(FactoMineR))

# Set work directory
setwd("C:/data/")

# Loading data
full_data <- read.csv("output-simple.csv", header = TRUE, sep = ',', dec = '.')

#################################
# 2. Descriptive Data Analysis  #
#################################

# Show dataframe dim(rows, cols)
dim(full_data)

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
nCV <- 5

# K-folds number
nFolds <- 5

# Variables to store the detection of Label=subjective
detection.subj.svm.lin <- rep(0, nCV)
detection.subj.svm.rad <- rep(0, nCV)
detection.subj.svm.pol <- rep(0, nCV)
detection.subj.svm.sig <- rep(0, nCV)

# Variables to store the detection of Label=objective
detection.obj.svm.lin <- rep(0, nCV)
detection.obj.svm.rad <- rep(0, nCV)
detection.obj.svm.pol <- rep(0, nCV)
detection.obj.svm.sig <- rep(0, nCV)

# Variables to store the average global error
error.global.svm.lin <- rep(0, nCV)
error.global.svm.rad <- rep(0, nCV)
error.global.svm.pol <- rep(0, nCV)
error.global.svm.sig <- rep(0, nCV)

# Start cross-validation
for(i in 1:nCV) {
  cat("ix", i)
  
  # Groups are created (for this case 5)
  groups <- createFolds(1:nRows, k = nFolds)
  
  # Internal counters of the prediction of the subjective
  subj.svm.lin <- 0
  subj.svm.rad <- 0
  subj.svm.pol <- 0
  subj.svm.sig <- 0
  
  # Internal counters of the prediction of the objective
  obj.svm.lin <- 0
  obj.svm.rad <- 0
  obj.svm.pol <- 0
  obj.svm.sig <- 0
  
  # Internal error counters of the models
  error.svm.lin <- 0
  error.svm.rad <- 0
  error.svm.pol <- 0
  error.svm.sig <- 0
  
  # Loop to perform the cross-validation
  for(k in 1:nFolds) {
    
    # The learning and testing datasets are generated
    sample <- groups[[k]]
    tlearning <- full_data[-sample, ]
    ttesting <- full_data[sample, ]

    # 1. A model is generated with the Vector Support Machines method
    cat("- svm linear, ix:", i, ", k:", k, "\n")
    model <- svm(Label~., data = tlearning, kernel = "linear", gama=0.25, cost=1)
    prediction <- predict(model, ttesting)
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.svm.lin <- subj.svm.lin + MC[2, 2]
    obj.svm.lin <- obj.svm.lin + MC[1, 1]
    error.svm.lin <- error.svm.lin + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 2. A model is generated with the Vector Support Machines method
    cat("- svm radial, ix:", i, ", k:", k, "\n")
    model <- svm(Label~., data = tlearning, kernel="radial", gama=0.25, cost=1)
    prediction <- predict(model, ttesting)
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.svm.rad <- subj.svm.rad + MC[2, 2]
    obj.svm.rad <- obj.svm.rad + MC[1, 1]
    error.svm.rad <- error.svm.rad + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 3. A model is generated with the Vector Support Machines method
    cat("- svm polynomial, ix:", i, ", k:", k, "\n")
    model <- svm(Label~., data = tlearning, kernel="polynomial", gama=0.25, cost=1)
    prediction <- predict(model, ttesting)
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.svm.pol <- subj.svm.pol + MC[2, 2]
    obj.svm.pol <- obj.svm.pol + MC[1, 1]
    error.svm.pol <- error.svm.pol + (1 - (sum(diag(MC)) / sum(MC))) * 100
    
    # 4. A model is generated with the Vector Support Machines method
    cat("- svm sigmoid, ix:", i, ", k:", k, "\n")
    model <- svm(Label~., data = tlearning, kernel="sigmoid", gama=0.25, cost=1)
    prediction <- predict(model, ttesting)
    actual <- ttesting[,ixTypeVar]
    MC <- table(actual, prediction)
    
    # The quality of the model is saved
    subj.svm.sig <- subj.svm.sig + MC[2, 2]
    obj.svm.sig <- obj.svm.sig + MC[1, 1]
    error.svm.sig <- error.svm.sig + (1 - (sum(diag(MC)) / sum(MC))) * 100
  }
  
  # The amount of subjective detected by each method is saved
  detection.subj.svm.lin[i] <- subj.svm.lin
  detection.subj.svm.rad[i] <- subj.svm.rad
  detection.subj.svm.pol[i] <- subj.svm.pol
  detection.subj.svm.sig[i] <- subj.svm.sig
  
  # The amount of objective detected by each method is saved
  detection.obj.svm.lin[i] <- obj.svm.lin
  detection.obj.svm.rad[i] <- obj.svm.rad
  detection.obj.svm.pol[i] <- obj.svm.pol
  detection.obj.svm.sig[i] <- obj.svm.sig
  
  # The average global error is saved for each model
  error.global.svm.lin[i] <- error.svm.lin / nFolds
  error.global.svm.rad[i] <- error.svm.rad / nFolds
  error.global.svm.pol[i] <- error.svm.pol / nFolds
  error.global.svm.sig[i] <- error.svm.sig / nFolds
}

###################################
# 5. Plotting the Models Results  #
###################################

# The limits for Plot 1 are calculated
yLim <- c(min(error.global.svm.lin, error.global.svm.rad, error.global.svm.pol, error.global.svm.sig) * 0.9,
          max(error.global.svm.lin, error.global.svm.rad, error.global.svm.pol, error.global.svm.sig) * 1.3)

# The curves are plotted with the average global error of each model
plot(error.global.svm.lin, col = "magenta", type = "b", ylim = yLim, main = "Models error for subjectivity analysis",
     xlab = "Número de iteración", ylab = "Error Global", cex.axis = 0.5)
points(error.global.svm.rad, col = "blue", type = "b")
points(error.global.svm.pol, col = "lightblue3", type = "b")
points(error.global.svm.sig, col = "olivedrab", type = "b")

# The legend is added
legend("topright", legend = c("SVM Lin", "SVM Rad", "SVM Poly", "SVM Sigmoid"),
       col = c("magenta", "blue", "red", "lightblue3"),
       lty = 1, lwd = 2, ncol = 4, cex = 0.45)

# The limits for Plot 2 are setted
yLim <- c(nSubj * 0.5, nSubj * 1.05)

# The curves of the subjective detection are plotted
plot(detection.subj.svm.lin, col = "magenta", type = "b", ylim = yLim, main = "Correct detection of subjective articles",
     xlab = "Número de iteración", ylab = "Cantidad de SI detectados", cex.axis = 0.5)
points(detection.subj.svm.rad, col = "blue", type = "b")
points(detection.subj.svm.pol, col = "lightblue3", type = "b")
points(detection.subj.svm.sig, col = "olivedrab", type = "b")

# The legend is added
legend("topright", legend = c("SVM Lin", "SVM Rad", "SVM Poly", "SVM Sigmoid"),
       col = c("magenta", "blue", "red", "lightblue3"),
       lty = 1, lwd = 2, ncol = 4, cex = 0.45)

# The limits for Plot 3 are setted
yLim <- c(nObj * 0.5, nObj * 1.05)

# The curves of the objective detection are plotted
plot(detection.obj.svm.lin, col = "magenta", type = "b", ylim = yLim, main = "Correct detection of objective articles",
     xlab = "Número de iteración", ylab = "Cantidad de NO detectados", cex.axis = 0.5)
points(detection.obj.svm.rad, col = "blue", type = "b")
points(detection.obj.svm.pol, col = "lightblue3", type = "b")
points(detection.obj.svm.sig, col = "olivedrab", type = "b")

# The legend is added
legend("topright", legend = c("SVM Lin", "SVM Rad", "SVM Poly", "SVM Sigmoid"),
       col = c("magenta", "blue", "red", "lightblue3"),
       lty = 1, lwd = 2, ncol = 4, cex = 0.45)

# The average Errors of the models are shown
print(mean(error.global.svm.lin))
print(mean(error.global.svm.rad))
print(mean(error.global.svm.pol))
print(mean(error.global.svm.sig))

# The subjective detected by each models are shown
print(mean(detection.subj.svm.lin))
print(mean(detection.subj.svm.rad))
print(mean(detection.subj.svm.pol))
print(mean(detection.subj.svm.sig))

# The objective detected by each models are shown
print(mean(detection.obj.svm.lin))
print(mean(detection.obj.svm.rad))
print(mean(detection.obj.svm.pol))
print(mean(detection.obj.svm.sig))

# End of the ML process
ptm <- (proc.time() - ptm)
cat("Time: ", ptm)

#####################
# 6. End of Script  #
#####################