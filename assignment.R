## @knitr LoadingLibraries


library(caret)
library(randomForest)


## @knitr StartClustering


library(doParallel)
cluster <- makeCluster(detectCores())
registerDoParallel(cluster)


## @knitr ClusteringWorkaround


# Probably don't need the workaround discussed in this post:
# http://stackoverflow.com/questions/24786089/parrf-on-caret-not-working-for-more-than-one-core
####cluster <- makePSOCKcluster(4)
####clusterEvalQ(cluster, library(foreach))


## @knitr ExploratoryFull


# Lots of NAs
# sapply(fullTraining, function(x) sum(is.na(x)))


## @knitr LoadSplitData


# The data files are pre-downloaded and saved in the "data" folder
trainingFile <- "data/pml-training.csv"
testingFile <- "data/pml-testing.csv"

fullTraining <- read.csv(trainingFile, na.strings=c("NA", "#DIV/0!"))
submission <- read.csv(testingFile, na.strings=c("NA", "#DIV/0!"))

set.seed(27182818)
inTrain <- createDataPartition(y=fullTraining$classe,p=0.6,list=FALSE)

cols <- grep("^(roll|pitch|yaw|total_accel|gyros|accel|magnet)_(belt|arm|forearm|dumbbell)(_(x|y|z))?",
             colnames(fullTraining), perl=TRUE, value=TRUE)
training <- fullTraining[inTrain, c("classe", cols)]
testing <- fullTraining[-inTrain, c("classe", cols)]


## @knitr SmallSampleForExperiments


trainingQuick <- training[sample(1:nrow(training), 1000),]


## @knitr Exploratory


# Look at a quick-and-dirty random forest so we can find a reasonble
# number of trees to speed it up the ultimate model training
fitRandomForest <- randomForest(as.factor(classe) ~ ., data=training, ntree=100)


## @knitr ExploratoryRFPlot


# Look at OOB error vs number of trees
plot(fitRandomForest, main="Random Forest Error vs. # of Trees")


## @knitr NotUsedExploratory


## in-sample
#pred <- predict(fitRandomForest, training)
#confusionMatrix(pred, training$classe)
## out-of-sample
#pred <- predict(fitRandomForest, testing)
#confusionMatrix(pred, testing$classe)
##
#predict(fitRandomForest, submission, type="response")


# plot(training$yaw_belt, col=training$classe)


## @knitr Parameters


# Set seeds for repeatable runs using parallel computation.
# 10xrepeated CV for 5 repeats == 50, so we need 50+1 == 51 sets of seeds in our vector.
# If we have too many seeds, it's not a problem.  The extras are ignored.
seeds <- vector(mode = "list", length = 51)
for(i in 1:50) seeds[[i]] <- sample.int(999999999, 25)

# For the last model:
seeds[[51]] <- sample.int(999999999, 1)

# Use the same cross-validation folding for most fits
trControl <- trainControl(method="repeatedcv", repeats=5, seeds=seeds, allowParallel=TRUE)


## @knitr ModelOne


# Random Forest
tGrid <- data.frame(mtry=length(cols)/2) # Factor of 3 speed up by providing a specific value to use
model1Time <- system.time(model1 <- train(as.factor(classe) ~ ., data=training, method="rf",
                                          ntree=40, prox=TRUE, tuneGrid=tGrid, trControl=trControl))
# in-sample error
model1_ISError <- confusionMatrix(predict(model1, training), training$classe)
# out-of-sample
model1_OOSError <- confusionMatrix(predict(model1, testing), testing$classe)


## @knitr ModelTwo


# k nearest neighbors
model2Time <- system.time(model2 <- train(classe ~ ., data=training, method="knn", tuneLength=20,
                                          trControl=trControl))
# in-sample error
model2_ISError <- confusionMatrix(predict(model2, training), training$classe)
# out-of-sample
model2_OOSError <- confusionMatrix(predict(model2, testing), testing$classe)


## @knitr ModelThree


# CART
model3Time <- system.time(model3 <- train(as.factor(classe) ~ ., data=training, method="rpart", trControl=trControl))
# in-sample error
model3_ISError <- confusionMatrix(predict(model3, training), training$classe)
# out-of-sample
model3_OOSError <- confusionMatrix(predict(model3, testing), testing$classe)


## @knitr ModelThreePlot


library(rattle)
fancyRpartPlot(model3$finalModel)


## @knitr ModelFour


# Stochastic Gradient Boosting
model4Time <- system.time(model4 <- train(classe ~ ., data=training, method="gbm", verbose=FALSE, trControl=trControl))
# in-sample error
model4_ISError <- confusionMatrix(predict(model4, training), training$classe)
# out-of-sample
model4_OOSError <- confusionMatrix(predict(model4, testing), testing$classe)


## @knitr ModelFive


# Support Vector Machines with Linear Kernel
model5Time <- system.time(model5 <- train(as.factor(classe) ~ ., data=training, method="svmLinear", trControl=trControl))
# in-sample error
model5_ISError <- confusionMatrix(predict(model5, training), training$classe)
# out-of-sample
model5_OOSError <- confusionMatrix(predict(model5, testing), testing$classe)


## @knitr ModelSix


# Linear Discriminant Analysis
model6Time <- system.time(model6 <- train(as.factor(classe) ~ ., data=training, method="lda", trControl=trControl))
# in-sample error
model6_ISError <- confusionMatrix(predict(model6, training), training$classe)
# out-of-sample
model6_OOSError <- confusionMatrix(predict(model6, testing), testing$classe)


## @knitr ModelNotUsed

# This model wasn't used in the final analysis simply because it isn't implemented in Caret

library(kernlab)
system.time(svmFit <- ksvm(classe ~ ., data=training, kernel="laplacedot", C=50, cross=5))
svmPred <- predict(svmFit, testing, type="response")
confusionMatrix(svmPred, testing$classe)
predict(svmFit, submission, type="response")


## @knitr StopClustering


stopCluster(cluster)
registerDoSEQ()


## @knitr PostRunCalculations


# Cross Validation estimated out-of-sample accuracy and error
# Models 2-4 report many accuracies and choose the most accurate model as the final model
model1EstAcc <- round(model1$results$Accuracy * 100, 1)
model2EstAcc <- round(max(model2$results$Accuracy * 100), 1)
model3EstAcc <- round(max(model3$results$Accuracy * 100), 1)
model4EstAcc <- round(max(model4$results$Accuracy * 100), 1)
model5EstAcc <- round(model5$results$Accuracy * 100, 1)
model6EstAcc <- round(model6$results$Accuracy * 100, 1)

model1EstErr <- 100 - model1EstAcc
model2EstErr <- 100 - model2EstAcc
model3EstErr <- 100 - model3EstAcc
model4EstErr <- 100 - model4EstAcc
model5EstErr <- 100 - model5EstAcc
model6EstErr <- 100 - model6EstAcc

# In-sample accuracy and error (measured on the training sample, of course)
model1ISAcc <- round(model1_ISError$overall[1] * 100, 1)
model2ISAcc <- round(model2_ISError$overall[1] * 100, 1)
model3ISAcc <- round(model3_ISError$overall[1] * 100, 1)
model4ISAcc <- round(model4_ISError$overall[1] * 100, 1)
model5ISAcc <- round(model5_ISError$overall[1] * 100, 1)
model6ISAcc <- round(model6_ISError$overall[1] * 100, 1)

model1ISErr <- 100 - model1ISAcc
model2ISErr <- 100 - model2ISAcc
model3ISErr <- 100 - model3ISAcc
model4ISErr <- 100 - model4ISAcc
model5ISErr <- 100 - model5ISAcc
model6ISErr <- 100 - model6ISAcc

# Out-of-sample accuracy and error as measured on the test sample
model1Acc <- round(model1_OOSError$overall[1] * 100, 1)
model2Acc <- round(model2_OOSError$overall[1] * 100, 1)
model3Acc <- round(model3_OOSError$overall[1] * 100, 1)
model4Acc <- round(model4_OOSError$overall[1] * 100, 1)
model5Acc <- round(model5_OOSError$overall[1] * 100, 1)
model6Acc <- round(model6_OOSError$overall[1] * 100, 1)

model1Err <- 100 - model1Acc
model2Err <- 100 - model2Acc
model3Err <- 100 - model3Acc
model4Err <- 100 - model4Acc
model5Err <- 100 - model5Acc
model6Err <- 100 - model6Acc

# How accurate was the cross-validated measurement of out-of-sample error?
model1cvErr <- round((abs(100 * ((1 - model1$results$Accuracy) - (1 - model1_OOSError$overall[1])))), 2)
model2cvErr <- round((abs(100 * ((1 - max(model2$results$Accuracy)) - (1 - model2_OOSError$overall[1])))), 2)
model3cvErr <- round((abs(100 * ((1 - max(model3$results$Accuracy)) - (1 - model3_OOSError$overall[1])))), 2)
model4cvErr <- round((abs(100 * ((1 - max(model4$results$Accuracy)) - (1 - model4_OOSError$overall[1])))), 2)
model5cvErr <- round((abs(100 * ((1 - model5$results$Accuracy) - (1 - model5_OOSError$overall[1])))), 2)
model6cvErr <- round((abs(100 * ((1 - model6$results$Accuracy) - (1 - model6_OOSError$overall[1])))), 2)


## @knitr VariableImportance


plot(varImp(model1), main = "Variable Importance", top = 52)

