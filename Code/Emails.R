if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")

library(readr)
library(tidyverse)
library(caret)

# loading names file
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names", dl)

# choosing lines that contains the names
lines<-read_lines(file = dl, skip_empty_rows = T)[34:90]

# choosing the names only (not the type)
lines<-sapply(lines,function(l){
  x<-str_sub(l,start = 2L)[1]
  str_split_fixed(x,":",2)[,1]
})
lines<-unname(lines)

# adding the last field name
names<-c( lines,"spam")


# loading the data file
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", dl)

# putting it all together (data & names)
emails<-read.delim2(dl,header = F, sep = ",", dec = ".") %>%
  set_names(., nm = names) %>% mutate(spam=ifelse(spam==1,"Spam", "Not Spam")) %>%
  mutate(spam=factor(spam, levels = c("Spam", "Not Spam"))) %>% 
  mutate(capital_run_length_longest=as.double(capital_run_length_longest)) %>%
  mutate(capital_run_length_total=as.double(capital_run_length_total)) %>%
  as.tibble()

# number of instances
nrow(emails)

# number of instances for Spam / Not Spam emails
table(emails$spam)


# plotting the distributions of some variables
emails %>% 
  pivot_longer(names_to = "Attr", values_to="values", cols=-spam) %>%
  filter(Attr %in% c("word_freq_000","word_freq_your",
                     "capital_run_length_total","char_freq_!")) %>%
  ggplot(aes(spam,values, fill=spam))+
  geom_boxplot()+
  facet_wrap(~Attr, scales = "free")+
  theme(axis.text.x = element_blank())

##############

# setting seed so that we all get the same numbers
set.seed(1,sample.kind = "Rounding")

# Partioning the dataset into test and training datasets
test_index<-createDataPartition(emails$spam, times = 1, p = 0.2, list = F)
train_set<-emails[-test_index,]
test_set<-emails[test_index,]

nrow(train_set)

# spliting the training dataset for convenience 
x<-train_set %>% select(-spam) %>% as.matrix()
y=train_set$spam


##############
# GLM model

# no tuning parameters
modelLookup("glm")

control <- trainControl(summaryFunction = twoClassSummary) 
# training with specificity as the metric, not Accuracy
train_glm<-train(x,y, method="glm",  metric = "Spec", trControl=control )
# predicting on test dataset
y_hat_glm<-predict(train_glm, newdata = test_set)
# performance
confusionMatrix(y_hat_glm, test_set$spam)$table
specificity(y_hat_glm, test_set$spam)
# creating performance tracker
specificity_df<-data.frame(method="glm", Spec=specificity(y_hat_glm, test_set$spam),
                           Accuracy=as.double(confusionMatrix(y_hat_glm, test_set$spam)$overall["Accuracy"]))
specificity_df
# viewing important variables
varImp(train_glm)

##############
# Naive Bayes model

# three tuning parameters
modelLookup("naive_bayes")

# training with specificity as the metric, not Accuracy, with default tuning parameters
train_naive_bayes<-train(x,y, method="naive_bayes",  metric = "Spec", trControl=control )
# predicting on test dataset
y_hat_naive_bayes<-predict(train_naive_bayes, newdata = test_set)
# performance
confusionMatrix(y_hat_naive_bayes, test_set$spam)$table
specificity(y_hat_naive_bayes, test_set$spam)
# adding to performance tracker
specificity_df<-rbind(specificity_df,data.frame(method="naive_bayes", Spec=specificity(y_hat_naive_bayes, test_set$spam),
                                                Accuracy=as.double(confusionMatrix(y_hat_naive_bayes, test_set$spam)$overall["Accuracy"])))
specificity_df
# viewing important variables
varImp(train_naive_bayes)

############
# LDA model

# no tuning parameters
modelLookup("lda")

# training with specificity as the metric, not Accuracy
train_lda<-train(x,y, method="lda", metric = "Spec", trControl=control )
# predicting on test dataset
y_hat_lda<-predict(train_lda, newdata = test_set)
# performance
confusionMatrix(y_hat_lda, test_set$spam)$table
specificity(y_hat_lda, test_set$spam)
# adding to performance tracker
specificity_df<-rbind(specificity_df,data.frame(method="lda", Spec=specificity(y_hat_lda, test_set$spam),
                                                Accuracy=as.double(confusionMatrix(y_hat_lda, test_set$spam)$overall["Accuracy"])))
specificity_df

# viewing important variables
varImp(train_lda)


############
# KNN model

# one tuning parameter, k
modelLookup("knn")

# creating a tuning grid for k
grid <- data.frame(k = seq(2, 20))

# training with specificity as the metric, not Accuracy, with prepared tuning grid
train_knn<-train(x,y, method="knn", metric = "Spec", trControl=control , tuneGrid = grid)
# checking the performance of the ks
ggplot(train_knn, highlight = T)

# predicting on test dataset
y_hat_knn<-predict(train_knn, newdata = test_set)
# performance
confusionMatrix(y_hat_knn, test_set$spam)$table
specificity(y_hat_knn, test_set$spam)
# adding to performance tracker
specificity_df<-rbind(specificity_df,data.frame(method="knn", Spec=specificity(y_hat_knn, test_set$spam),
                                                Accuracy=as.double(confusionMatrix(y_hat_knn, test_set$spam)$overall["Accuracy"])))
specificity_df

# viewing important variables
varImp(train_knn)

############
# RPART model

# one tuning parameter, cp
modelLookup("rpart")

# creating a tuning grid for cp
grid <- data.frame(cp=seq(0,0.2, length.out = 21))

# training with specificity as the metric, not Accuracy, with prepared tuning grid
train_rpart<-train(x,y, method="rpart", metric = "Spec", trControl=control , tuneGrid = grid)
# checking the performance of the cps
ggplot(train_rpart, highlight = T)

# plotting the final model tree
plot(train_rpart$finalModel,margin = 0.1)
text(train_rpart$finalModel, cex=0.75)

# predicting on test dataset
y_hat_rpart<-predict(train_rpart, newdata = test_set)
# performance
confusionMatrix(y_hat_rpart, test_set$spam)$table
specificity(y_hat_rpart, test_set$spam)
# adding to performance tracker
specificity_df<-rbind(specificity_df,data.frame(method="rpart", Spec=specificity(y_hat_rpart, test_set$spam),
                                                Accuracy=as.double(confusionMatrix(y_hat_rpart, test_set$spam)$overall["Accuracy"])))
specificity_df

# viewing important variables
varImp(train_rpart)

############
# Random Forest model

# one tuning parameter, mtry
modelLookup("rf")

# training with specificity as the metric, not Accuracy, with a specific tuning grid
train_rf<-train(x,y, method="rf", metric = "Spec", trControl=control ,
                tuneGrid = data.frame(mtry=c(3, 5, 7, 9)))
# checking the performance of the mtrys
ggplot(train_rf, highlight = T)

# predicting on test dataset
y_hat_rf<-predict(train_rf, newdata = test_set)
# performance
confusionMatrix(y_hat_rf, test_set$spam)$table
specificity(y_hat_rf, test_set$spam)
# adding to performance tracker
specificity_df<-rbind(specificity_df,data.frame(method="rf", Spec=specificity(y_hat_rf, test_set$spam),
                                                Accuracy=as.double(confusionMatrix(y_hat_rf, test_set$spam)$overall["Accuracy"])))
specificity_df

# viewing important variables
varImp(train_rf)

############
# Bagged Tree model

# no tuning parameters
modelLookup("treebag")

# training with specificity as the metric, not Accuracy
train_treebag<-train(x,y, method="treebag",  metric = "Spec", trControl=control )

# predicting on test dataset
y_hat_treebag<-predict(train_treebag, newdata = test_set)
# performance
confusionMatrix(y_hat_treebag, test_set$spam)$table
specificity(y_hat_treebag, test_set$spam)
# adding to performance tracker
specificity_df<-rbind(specificity_df,data.frame(method="treebag", Spec=specificity(y_hat_treebag, test_set$spam),
                                                Accuracy=as.double(confusionMatrix(y_hat_treebag, test_set$spam)$overall["Accuracy"])))
specificity_df

# viewing important variables
varImp(train_treebag)

############
# Ensemble model

# creating a dataframe for all previous predictions
ensemble<-data.frame(y_hat_glm,y_hat_naive_bayes,y_hat_lda,
                     y_hat_knn,y_hat_rpart,y_hat_rf,y_hat_treebag)
# defining the prediction of ensemble
y_hat_ensemble <-apply(ensemble,1 , function(x) ifelse(mean(x=="Spam")>0.5,"Spam","Not Spam"))
y_hat_ensemble<-factor(y_hat_ensemble, levels = c("Spam","Not Spam"))

# Performance
confusionMatrix(y_hat_ensemble, test_set$spam)$table
sensitivity(y_hat_ensemble, test_set$spam)
# adding to performance tracker
specificity_df<-rbind(specificity_df,data.frame(method="ensemble", Spec=specificity(y_hat_ensemble, test_set$spam),
                                                Accuracy=as.double(confusionMatrix(y_hat_ensemble, test_set$spam)$overall["Accuracy"])))
specificity_df

# saving data for easy access in the report
save.image("Emails.RData")
