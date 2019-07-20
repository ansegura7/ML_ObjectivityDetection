###############################################################################################################
# Course: MINE-4205 - Knowledge Discovery from Social and Information Networks                                #
# Title: Implementation for objectivity detection                                                             #
# Authors: Segura Tinoco, German Andres and Cuevas Saavedra, Vladimir Enrique                                 #
# Date: July 14, 2018                                                                                         #
# Experiment: 2.b                                                                                               #
###############################################################################################################

# Start process
rm(list = ls(all = TRUE))
ptm <- proc.time()

# Loading library
library(SSL)

# Set work directory
setwd("C:/Data/")

# Loading data (View 1)
dateset <- read.csv("output-stemmed.csv", header = TRUE, sep = ',', dec = '.')
nCol <- ncol(dateset)
nRows <- nrow(dateset)

# Get Objective/Subjective proportion
nSubj <- sum(dateset$Label == "subjective")
nObj <- nRows - nSubj
nObjClass <- nObj * 100 / nRows
cat("# Objective: ", nObj, ", # Subjective: ", nSubj, ", % Objective Class: ", nObjClass)

# 1 objective, 2 subjective
x <- dateset[,1:(nCol-1)]
y <- c(rep(1, nObj), rep(2, nSubj))

# Experiments
perc <- array(c(0.1, 0.2, 0.3, 0.4, 0.5))
nIter <- length(perc)

# Counters
detection.obj <- rep(0, nIter)
detection.subj <- rep(0, nIter)
error.global <- rep(0, nIter)

for(i in 1:nIter) {
  
  # Counters
  obj.count <- 0
  subj.count <- 0
  error.count <- 0
  
  # Set bounds
  topObj <- as.integer(nObj * perc[i])
  topSub <- as.integer(nObj + 1 + nSubj * perc[i])
  known.label <- c(1:topObj, (nObj+1):topSub)
  
  # Set sub-datasets
  xl <- x[known.label,]
  xu <- x[-known.label,]
  yl <- y[known.label]
  yu <- y[-known.label]
  nLearn <- as.integer(length(yu) * 0.1)
  
  # Co-Training and Make prediction
  predict <- sslCoTrain(xl, yl, xu, method1="xgb", nrounds1 = 100, method2="xgb", nrounds2 = 100, n=nLearn)
  #predict <- sslCoTrain(xl, yl, xu, method1="nb", method2="nb", n=nLearn)
  MC <- table(yu, predict)
  
  # Saving results
  detection.obj[i] <- MC[1, 1]
  detection.subj[i] <- MC[2, 2]
  error.global[i] <- (1 - (sum(diag(MC)) / sum(MC))) * 100
  cat('topObj:', topObj, 'topSub:', topSub, 'length:', 'xl:', nrow(xl), 'xu:', nrow(xu), 'yl:', length(yl), 'yu:', length(yu), 'nLearn:', nLearn, 'error:', error.global[i], '\n')
  print(MC)
}

# Ploting results
print(mean(error.global))
print(error.global)
# plot(error.global, col = "blue", type = "b", ylim = c(0, 100), main = "Co-Training error", xlab = "% Training data", ylab = "Error", cex.axis = 0.5)

# Ploting results
print(mean(detection.obj))
print(detection.obj)
# plot(detection.obj, col = "blue", type = "b", ylim = c(0, 100), main = "Co-Training error", xlab = "% Training data", ylab = "Error", cex.axis = 0.5)

# Ploting results
print(mean(detection.subj))
print(detection.subj)
# plot(detection.subj, col = "blue", type = "b", ylim = c(0, 100), main = "Co-Training error", xlab = "% Training data", ylab = "Error", cex.axis = 0.5)

# End of process
ptm <- (proc.time() - ptm)
cat("Time: ", ptm, "\n")

####################
# 5. End of Script #
####################