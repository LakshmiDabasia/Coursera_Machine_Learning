#load data
data<-read.csv("pml-training.csv")
validation<-read.csv("pml-testing.csv")
#prepare data
inTrain<-createDataPartition(data$classe, p=0.6, list=FALSE)
training<-data[inTrain,]
testing<-data[-inTrain,]

#install and load pacakges
install.packages("caret")
install.packages("gbm")
library(caret)
library(randomForest)
library(plyr)
library(gbm)
library(caret); library(rattle); library(rpart); library(rpart.plot)
library(randomForest)
library("e1071")
#EDA
summary(training)
#remove excessive na columns and columns with nearzerovar
reduced <- training[ lapply( training, function(x) sum(is.na(x)) / length(x) ) < 0.1]
reduced <-reduced[-nearZeroVar(reduced)]
reduced <-reduced[,6:ncol(reduced)]
reduced<-reduced[,-ncol(reduced)]
reduced <- data.frame(sapply(reduced, function(x) as.numeric(as.character(x))))
#remove highly correlated variables
corMatrix<-cor(reduced)
removelist<-findCorrelation(corMatrix, cutoff=0.8, verbose=FALSE, names=TRUE)
removelist<-as.vector(removelist)
reduced<-reduced[,!(names(reduced) %in% removelist)]

#subset data to reduced from
training<-training[,c(colnames(reduced),"classe")]
testing<-testing[,c(colnames(reduced),"classe")]
validation<-validation[,c(colnames(reduced))]

#building models
#random forest
rf_mod<-randomForest(classe~., data=training)
rf_mod
rf_predicted<-predict(rf_mod, testing)
rf_conf<-confusionMatrix(testing$classe, rf_predicted)
#boosting
settings<-trainControl(method="repeatedcv", number=5,repeats = 1)
gmb_mod<-train(classe ~ ., data=training, method='gbm', verbose=FALSE, trcontrol=settings)
gbm_mod
gbm_predicted<-predict(gbm_mod, testing)
gbm_conf<-confusionMatrix(testing$classe, gbm_predicted)
#support vector machine
svm_mod<-svm(classe~., data=training)
svm_mod
svm_predicted<-predict(svm_mod, testing)
svm_conf<-confusionMatrix(testing$classe, svm_predicted)

svm_predicted
