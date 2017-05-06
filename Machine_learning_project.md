# Coursera Machine Learning Project
Lakshmi D  
1 May 2017  



## Introduction

This report looks at predicting how well an exercise was performed. The data used throughout is sourced from  http://groupware.les.inf.puc-rio.br/har. 

## Loading of Packages

```
## Warning: package 'caret' was built under R version 3.2.5
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.2.5
```

```
## Warning: package 'randomForest' was built under R version 3.2.5
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## Warning: package 'plyr' was built under R version 3.2.5
```

```
## Warning: package 'gbm' was built under R version 3.2.5
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.3
```

```
## Warning: package 'rattle' was built under R version 3.2.5
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```
## Warning: package 'rpart.plot' was built under R version 3.2.5
```

```
## Warning: package 'e1071' was built under R version 3.2.5
```

## Data Loading and Partioning


```r
#load data
data<-read.csv("pml-training.csv")
validation<-read.csv("pml-testing.csv")
#prepare data
inTrain<-createDataPartition(data$classe, p=0.6, list=FALSE)
training<-data[inTrain,]
testing<-data[-inTrain,]
```

## Explatory Data Analysis



```r
#EDA
#summary(training)
dim(training)
```

```
## [1] 11776   160
```

We notice here that the data set is extremely large and has some missing values.

## Pre-procesing and Dimension Reduction 
We first remove any columns with more than 10% of missing data. We then remove variables which have very little variance. Variables which are highly correlated are then also removed (keeping just one). 



## Building Models

```r
#random forest
rf_mod<-randomForest(classe~., data=training)
rf_mod
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 6
## 
##         OOB estimate of  error rate: 0.42%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3347    0    0    0    1 0.0002986858
## B    7 2267    5    0    0 0.0052654673
## C    0   12 2041    1    0 0.0063291139
## D    0    0   19 1909    2 0.0108808290
## E    0    0    0    2 2163 0.0009237875
```

```r
#support vector machine learning (SVM)
svm_mod<-svm(classe~., data=training)
svm_mod
```

```
## 
## Call:
## svm(formula = classe ~ ., data = training)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  radial 
##        cost:  1 
##       gamma:  0.025 
## 
## Number of Support Vectors:  5591
```

##Cross Validation

```r
#random forest validation
rf_predicted<-predict(rf_mod, testing)
rf_conf<-confusionMatrix(testing$classe, rf_predicted)
rf_conf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    1 1517    0    0    0
##          C    0    9 1359    0    0
##          D    0    0   11 1275    0
##          E    0    0    0    5 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9967          
##                  95% CI : (0.9951, 0.9978)
##     No Information Rate : 0.2846          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9958          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9941   0.9920   0.9961   1.0000
## Specificity            1.0000   0.9998   0.9986   0.9983   0.9992
## Pos Pred Value         1.0000   0.9993   0.9934   0.9914   0.9965
## Neg Pred Value         0.9998   0.9986   0.9983   0.9992   1.0000
## Prevalence             0.2846   0.1945   0.1746   0.1631   0.1832
## Detection Rate         0.2845   0.1933   0.1732   0.1625   0.1832
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9998   0.9970   0.9953   0.9972   0.9996
```

```r
#svm validation
svm_predicted<-predict(svm_mod, testing)
svm_conf<-confusionMatrix(testing$classe, svm_predicted)
svm_conf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2202    7   22    0    1
##          B  104 1373   32    3    6
##          C    1   49 1303   13    2
##          D    2    1  133 1149    1
##          E    0    3   26   43 1370
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9428          
##                  95% CI : (0.9374, 0.9478)
##     No Information Rate : 0.2943          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9275          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9537   0.9581   0.8595   0.9512   0.9928
## Specificity            0.9946   0.9774   0.9897   0.9794   0.9889
## Pos Pred Value         0.9866   0.9045   0.9525   0.8935   0.9501
## Neg Pred Value         0.9809   0.9905   0.9671   0.9910   0.9984
## Prevalence             0.2943   0.1826   0.1932   0.1540   0.1759
## Detection Rate         0.2807   0.1750   0.1661   0.1464   0.1746
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9741   0.9678   0.9246   0.9653   0.9908
```

We see that the random forest model has the least out-of-bag error.
As a result, this model was selected to use on the final testing data provided.

##Final Test

```r
rf_validation_predicted<-predict(rf_mod, validation)
rf_validation_predicted
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
