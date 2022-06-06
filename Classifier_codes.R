#load libraries 
library(pastecs)
library(tree)
library(rpart)
library(e1071)
library(adabag)
library(randomForest)
library(ROCR)
library(sf)
library(skimr)


rm(list = ls())
WAUS <- read.csv("WarmerTomorrow2022.csv", stringsAsFactors = T)
L <- as.data.frame(c(1:49))
set.seed(31842305) # Your Student ID is the random seed
L <- L[sample(nrow(L), 10, replace = FALSE),] # sample 10 locations
WAUS <- WAUS[(WAUS$Location %in% L),]
WAUS <- WAUS[sample(nrow(WAUS), 2000, replace = FALSE),] # sample 2000 rows

#Omit rows which are not needed for statistic description
df_stat_sum <-  subset(WAUS, select = -c(1,2,3,4,10,12,13,24))
skim(df_stat_sum)



#Omit rows with NA and set target variable to factor
WAUS = na.omit(WAUS)
WAUS$WarmerTomorrow = as.factor(WAUS$WarmerTomorrow)


###############Question 1###################

warm_tmr = as.data.frame(table(WAUS$WarmerTomorrow))

#calculate the proportion of warmer and cooler
warmer =  round((warm_tmr[2,2] / sum(warm_tmr[,2])) * 100, digits = 2) #52.43
cooler = round((warm_tmr[1,2] / sum(warm_tmr[,2])) * 100, digits = 2) # 47.57

###############Question 2###################

#This part was done above already


###############Question 3###################

#### Divide your data into a 70% training and 30% test

set.seed(31842305)
train.row = sample(1:nrow(WAUS), 0.7*nrow(WAUS))
train.data = WAUS[train.row,]
test.data = WAUS[-train.row,]

###############Question 4###################

#### Question: Implement a classification model using each of the following techniques

#Decision Tree 
tree.fit = tree(WarmerTomorrow ~., data = train.data, method = "class")

#Naïve Bayes
naive.fit = naiveBayes(WarmerTomorrow ~., data = train.data)

##Bagging 
wbag.fit = bagging(WarmerTomorrow ~., train.data, mfinal = 10)

#Boosting 
wboost.fit = boosting(WarmerTomorrow ~., train.data, mfinal = 10)

#Random Forest 
rf.fit = randomForest(WarmerTomorrow ~., train.data)


###############Question 5###################

#Question: Using the test data, classify each of the test cases as 'warmer tomorrow' or 'not warmer tomorrow'. Create a confusion matrix and report the accuracy of each model.

#Decision Tree
tree.pred = predict(tree.fit, test.data, type = "class")
#confusion matrix
tree.cfm = table(actual = test.data$WarmerTomorrow, predicted = tree.pred)
#accuracy of model
tree.acc = round(mean(tree.pred == test.data$WarmerTomorrow)*100, digits = 2)
cat("Decision Tree accuracy is: ", tree.acc, "%")



#Naïve Bayes
naive.pred = predict(naive.fit, test.data)
#confusion matrix
naive.cfm = table(actual = test.data$WarmerTomorrow, predicted = naive.pred)
#accuracy of model
naive.acc = round(mean(naive.pred == test.data$WarmerTomorrow)*100, digits = 2)
cat("Naïve Bayes model accuracy is: ", naive.acc, "%")



##Bagging 
wbag.fit_pred = predict.bagging(wbag.fit, test.data)
#confusion matrix
wbag.fit.cfm = wbag.fit_pred$confusion
#accuracy of model
wbag.fit.acc = round(mean(wbag.fit_pred$class == test.data$WarmerTomorrow)*100, digits = 2)
cat("Bagging ensemble model accuracy is: ", wbag.fit.acc, "%")



#Boosting
wboost.fit_pred = predict.boosting(wboost.fit, test.data)
#confusion matrix
wboost.fit.cfm = wboost.fit_pred$confusion
#accuracy of model
wboost.fit.acc = round(mean(wboost.fit_pred$class == test.data$WarmerTomorrow)*100, digits = 2)
cat("Boosting ensemble model accuracy is: ", wboost.fit.acc, "%")


#Random Forest 
rf.fit_pred = predict(rf.fit, test.data)
#confusion matrix
rf.fit.cfm = table(actual = test.data$WarmerTomorrow, predicted = rf.fit_pred)
#accuracy of model
rf.fit.acc = round(mean(rf.fit_pred == test.data$WarmerTomorrow)*100, digits = 2)
cat("Random Forest ensemble model accuracy is: ", rf.fit.acc, "%")


###############Question 6###################
#Question: Calculate confidence and construct ROC curve

#Decision Tree 
tree.conf = predict(tree.fit, test.data, type = "vector")
tree.conf.pred = prediction(tree.conf[,2], test.data$WarmerTomorrow)
tree.perf = performance(tree.conf.pred, "tpr", "fpr")
plot(tree.perf, col = "black") 
abline(0,1)



#Naïve Bayes
naive.conf = predict(naive.fit, test.data, type = "raw")
naive.conf.pred = prediction(naive.conf[,2], test.data$WarmerTomorrow)
naive.perf = performance(naive.conf.pred, "tpr", "fpr")
plot(naive.perf,add = TRUE, col = "blueviolet")


#Bagging 
bag.conf = prediction(wbag.fit_pred$prob[,2], test.data$WarmerTomorrow) 
bag.perf = performance(bag.conf, "tpr", "fpr") 
plot(bag.perf, col = "green", add = TRUE)


#Boosting 
boost.conf = prediction(wboost.fit_pred$prob[,2], test.data$WarmerTomorrow) 
boost.perf = performance(boost.conf, "tpr", "fpr")
plot(boost.perf, col ="red", add = TRUE) 


#Random Forest 
rf.conf = predict(rf.fit, test.data, type = "prob")
rf.conf.pred = prediction(rf.conf[,2], test.data$WarmerTomorrow)
rf.perf = performance(rf.conf.pred, "tpr", "fpr")
plot(rf.perf, col = "cyan", add = TRUE) 


#add legend to ROC plot
legend("topleft", legend = c("Decision Tree", "Naive Bayes", "Bagging", "Boosting", "Random Forest"),col = c("black","blueviolet","green","red","cyan"), cex = 0.8, lty = 1)


#Calculate the AUC 

#Decision Tree 
tree.auc = performance(tree.conf.pred, "auc") 
tree.auc = as.numeric(tree.auc@y.values)
cat("Decision Tree AUC is: ", tree.auc)

#Naive Bayes 
nv.auc = performance(naive.conf.pred, "auc")
nv.auc = as.numeric(nv.auc@y.values)
cat("Naive Bayes AUC is :", nv.auc)

#Bagging 
bag.auc = performance(bag.conf, "auc")
bag.auc = as.numeric(bag.auc@y.values)
cat("Bagging AUC is : ", bag.auc)

#Boosting 
boost.auc = performance(boost.conf, "auc")
boost.auc = as.numeric(boost.auc@y.values)
cat("Boosting AUC is: ", boost.auc)

#Random Forest 
rf.auc = performance(rf.conf.pred, "auc")
rf.auc = as.numeric(rf.auc@y.values)
cat("Random Forest is : ", rf.auc)


###############Question 8###################

#Decision Tree 
summary(tree.fit)

#Naive Bayes
#Cannot calculate as it is not a tree based  classifier. 

#Bagging 
bagging_importance = as.data.frame(as.table(wbag.fit$importance))

#Boosting 
boosting_importance = as.data.frame(as.table(wboost.fit$importance))

#Random Forest 
rf_importance = as.data.frame(as.table(rf.fit$importance))

#Remove variable "Rainfall" from train and test data. 
new_train.data = train.data[,-7]
new_test.data = test.data[, -7]

##### Fit the model to the new training and test data after removing variable "Rainfall"####

#Decision Tree
imp_tree.fit = tree(WarmerTomorrow ~., data =new_train.data, method = "class")
imp_tree.pred = predict(imp_tree.fit, new_test.data, type = "class")
#confusion matrix
imp_tree.cfm = table(actual = new_test.data$WarmerTomorrow, predicted = imp_tree.pred)
#accuracy of model
imp_tree.acc = round(mean(imp_tree.pred == new_test.data$WarmerTomorrow)*100, digits = 2)
cat("Decision Tree accuracy is: ", imp_tree.acc, "%")



##Bagging 
imp_wbag.fit = bagging(WarmerTomorrow ~., new_train.data, mfinal = 10)
imp_wbag.fit_pred = predict.bagging(imp_wbag.fit, new_test.data)
#confusion matrix
imp_wbag.fit.cfm = imp_wbag.fit_pred$confusion
#accuracy of model
imp_wbag.fit.acc = round(mean(imp_wbag.fit_pred$class == new_test.data$WarmerTomorrow)*100, digits = 2)
cat("Bagging ensemble model accuracy is: ", imp_wbag.fit.acc, "%")



#Boosting
imp_wboost.fit = boosting(WarmerTomorrow ~., new_train.data, mfinal = 10)
imp_wboost.fit_pred = predict.boosting(imp_wboost.fit, new_test.data)
#confusion matrix
imp_wboost.fit.cfm = imp_wboost.fit_pred$confusion
#accuracy of model
imp_wboost.fit.acc = round(mean(imp_wboost.fit_pred$class == new_test.data$WarmerTomorrow)*100, digits = 2)
cat("Boosting ensemble model accuracy is: ", imp_wboost.fit.acc, "%")


#Random Forest 
imp_rf.fit = randomForest(WarmerTomorrow ~., new_train.data)
imp_rf.fit_pred = predict(imp_rf.fit, new_test.data)
#confusion matrix
imp_rf.fit.cfm = tree.cfm = table(actual = new_test.data$WarmerTomorrow, predicted = imp_rf.fit_pred)
#accuracy of model
imp_rf.fit.acc = round(mean(imp_rf.fit_pred == new_test.data$WarmerTomorrow)*100, digits = 2)
cat("Random Forest ensemble model accuracy is: ", imp_rf.fit.acc, "%")



###############Question 9###################

#perform cross validation for our decision tree
test.fit = cv.tree(tree.fit, FUN = prune.misclass)
print(test.fit)

#prune down the tree size 
prune.tree.fit = prune.misclass(tree.fit, best = 4)
summary(prune.tree.fit)

#plot prune decision tree
plot(prune.tree.fit)
text(prune.tree.fit, pretty = 0)

#calculate accuracy of prune decision tree
ppredict = predict(prune.tree.fit, test.data, type="class")
table(actual = test.data$WarmerTomorrow, predicted = ppredict)
#accuracy = 59.2 

#calculate confidence and AUC of prune decision tree
prune.tree.conf = predict(prune.tree.fit, test.data, type = "vector")
prune.tree.pred = prediction(prune.tree.conf[,2], test.data$WarmerTomorrow)
ptree.auc = performance(prune.tree.pred, "auc")
ptree.auc = as.numeric(ptree.auc@y.values)
cat("Prune Decision Tree AUC is: ", ptree.auc)
#AUC = 0.624


###############Question 10###################
#Find original number of trees used
rf.fit$ntree #500 

#save training and testing data in new variable
imp.train = train.data
imp.test = test.data

#Algorithm to fit the number of trees from 501 to 700 by tuning different mtry value
for(ntrees in 500:700){
  set.seed(31842305) 
  new_rf.fit = randomForest(WarmerTomorrow ~., data = imp.train, importance = TRUE, ntree = ntrees, mtry=2)
  new_rf.pred = predict(new_rf.fit, imp.test)
  new_rf.cfm = table(actual = imp.test$WarmerTomorrow, predicted = new_rf.pred)
  new_rf.acc = round(mean(new_rf.pred == imp.test$WarmerTomorrow)*100, digits = 2)
  
  #calculate confidence and AUC of the new Random Forest model 
  new_rf.conf = predict(new_rf.fit, imp.test, type = "prob")
  new_rf.conf.pred = prediction(new_rf.conf[,2], imp.test$WarmerTomorrow)
  new_rf.auc = performance(new_rf.conf.pred, "auc")
  new_rf.auc = as.numeric(new_rf.auc@y.values)
  
  cat("Best Tree is: ", new_rf.fit$ntree, "accuracy is: ", new_rf.acc , "AUC value: ", new_rf.auc ,"\n")
  
} 


###############Question 11###################
#Package to train the classifier
install.packages("neuralnet")
library(neuralnet)
library(caret)

#Store train data and test data in new variable 
nn_train = train.data
nn_test = test.data

#one hot encoding for categorical variable
dummy <- dummyVars("~.", data = nn_train[,1:24])
dummy2 <- dummyVars("~.", data = nn_test[,1:24])
nn_train = data.frame(predict(dummy, newdata = nn_train))
nn_test = data.frame(predict(dummy, newdata = nn_test))

#Function for max-min normalise 
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

#Normalize the data using max-min method
ann_train = as.data.frame(lapply(nn_train, FUN = normalize))
ann_test = as.data.frame(lapply(nn_test, FUN = normalize))

#Fit the classifier
ws.nn = neuralnet(WarmerTomorrow.0 + WarmerTomorrow.1 ~., data = ann_train, hidden = 3 , linear.output = FALSE, threshold = 0.01, stepmax = 1e7)

#Visualize result
plot(ws.nn)

#Evaluate model performance 
ws_pred = compute(ws_nn, ann_test)
ws_pred = as.data.frame(round(ws_pred$net.result, 0))
table(actual = ann_test$WarmerTomorrow.1 , predicted = ws_pred$V1)
