if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")

library(readr)
library(tidyverse)
library(caret)

dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names", dl)

lines<-read_lines(file = dl, skip_empty_rows = T)[34:90]
lines<-sapply(lines,function(l){
  x<-str_sub(l,start = 2L)[1]
  str_split_fixed(x,":",2)[,1]
})
lines<-unname(lines)
names<-c( lines,"spam")

dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", dl)
emails<-read.delim2(dl,header = F, sep = ",", dec = ".") %>%
  set_names(., nm = names) %>% mutate(spam=ifelse(spam==1,"Spam", "Not Spam")) %>%
  mutate(spam=factor(spam, levels = c("Spam", "Not Spam"))) %>% 
  mutate(capital_run_length_longest=as.double(capital_run_length_longest)) %>%
  mutate(capital_run_length_total=as.double(capital_run_length_total)) %>%
  as.tibble()

nrow(emails)

table(emails$spam)

emails %>% pivot_longer(names_to = "Attr", values_to="values", cols=-spam) %>%
  ggplot(aes(spam,values, fill=spam))+
  geom_boxplot()+
  facet_wrap(~Attr, scales = "free")+
  theme(axis.text.x = element_blank())

##############

set.seed(1,sample.kind = "Rounding")

test_index<-createDataPartition(emails$spam, times = 1, p = 0.2, list = F)
train_set<-emails[-test_index,]
test_set<-emails[test_index,]
nrow(train_set)

x<-train_set %>% select(-spam) %>% as.matrix()
y=train_set$spam


##############


modelLookup("glm")

control <- trainControl(summaryFunction = twoClassSummary) 

train_glm<-train(x,y, method="glm",  metric = "Spec", trControl=control )
y_hat_glm<-predict(train_glm, newdata = test_set)
confusionMatrix(y_hat_glm, test_set$spam)$table
specificity(y_hat_glm, test_set$spam)
specificity_df<-data.frame(method="glm", Spec=specificity(y_hat_glm, test_set$spam))
specificity_df


##############


modelLookup("naive_bayes")

train_naive_bayes<-train(x,y, method="naive_bayes",  metric = "Spec", trControl=control )
y_hat_naive_bayes<-predict(train_naive_bayes, newdata = test_set)
confusionMatrix(y_hat_naive_bayes, test_set$spam)$table
specificity(y_hat_naive_bayes, test_set$spam)
specificity_df<-rbind(specificity_df,data.frame(method="naive_bayes", Spec=specificity(y_hat_naive_bayes, test_set$spam)))
specificity_df

varImp(train_naive_bayes)$importance[1:10,]

############

modelLookup("lda")

train_lda<-train(x,y, method="lda", metric = "Spec", trControl=control )
y_hat_lda<-predict(train_lda, newdata = test_set)
confusionMatrix(y_hat_lda, test_set$spam)$table
specificity(y_hat_lda, test_set$spam)
specificity_df<-rbind(specificity_df,data.frame(method="lda", Spec=specificity(y_hat_lda, test_set$spam)))
specificity_df

varImp(train_lda)$importance[1:10,]

############
# 
# modelLookup("qda")
# 
# train_qda<-train(x,y, method="qda", metric = "Spec", trControl=control )
# y_hat_qda<-predict(train_qda, newdata = test_set)
# confusionMatrix(y_hat_qda, test_set$spam)$table
# specificity(y_hat_qda, test_set$spam)
# specificity_df<-rbind(specificity_df,data.frame(method="qda", Spec=specificity(y_hat_qda, test_set$spam)))
# specificity_df
# 
# 

############

modelLookup("knn")

grid <- data.frame(k = seq(2, 20))

train_knn<-train(x,y, method="knn", metric = "Spec", trControl=control , tuneGrid = grid)
ggplot(train_knn, highlight = T)

y_hat_knn<-predict(train_knn, newdata = test_set)
confusionMatrix(y_hat_knn, test_set$spam)$table
specificity(y_hat_knn, test_set$spam)
specificity_df<-rbind(specificity_df,data.frame(method="knn", Spec=specificity(y_hat_knn, test_set$spam)))
specificity_df

varImp(train_knn)$importance[1:10,]
# 
# 
# ############
# 
# modelLookup("gamLoess")
# 
# grid <- expand.grid(span = seq(0.15, 0.65, len = 10), degree = 1)
# 
# train_loess<-train(x,y, method="gamLoess",  metric = "Spec", trControl=control , tuneGrid = grid)
# ggplot(train_loess, highlight = T)
# 
# y_hat_loess<-predict(train_loess, newdata = test_set)
# confusionMatrix(y_hat_loess, test_set$spam)$table
# specificity(y_hat_loess, test_set$spam)
# specificity_df<-rbind(specificity_df,data.frame(method="gamLoess", Spec=specificity(y_hat_loess, test_set$spam)))
# specificity_df

############

modelLookup("rpart")

grid <- data.frame(cp=seq(0,0.2, length.out = 21))

train_rpart<-train(x,y, method="rpart", metric = "Spec", trControl=control , tuneGrid = grid)
ggplot(train_rpart, highlight = T)

plot(train_rpart$finalModel,margin = 0.1)
text(train_rpart$finalModel, cex=0.75)

y_hat_rpart<-predict(train_rpart, newdata = test_set)
confusionMatrix(y_hat_rpart, test_set$spam)$table
specificity(y_hat_rpart, test_set$spam)
specificity_df<-rbind(specificity_df,data.frame(method="rpart", Spec=specificity(y_hat_rpart, test_set$spam)))
specificity_df


############

modelLookup("rf")

train_rf<-train(x,y, method="rf", metric = "Spec", trControl=control ,
                tuneGrid = data.frame(mtry=c(3, 5, 7, 9)))
ggplot(train_rf, highlight = T)

y_hat_rf<-predict(train_rf, newdata = test_set)
confusionMatrix(y_hat_rf, test_set$spam)$table
specificity(y_hat_rf, test_set$spam)
specificity_df<-rbind(specificity_df,data.frame(method="rf", Spec=specificity(y_hat_rf, test_set$spam)))
specificity_df



############

modelLookup("treebag")

train_treebag<-train(x,y, method="treebag",  metric = "Spec", trControl=control )

y_hat_treebag<-predict(train_treebag, newdata = test_set)
confusionMatrix(y_hat_treebag, test_set$spam)$table
specificity(y_hat_treebag, test_set$spam)
specificity_df<-rbind(specificity_df,data.frame(method="treebag", Spec=specificity(y_hat_treebag, test_set$spam)))
specificity_df


############


ensemble<-data.frame(y_hat_glm,y_hat_naive_bayes,y_hat_lda,
                     y_hat_knn,y_hat_rpart,y_hat_rf,y_hat_treebag)
y_hat_ensemble <-apply(ensemble,1 , function(x) ifelse(mean(x=="Spam")>0.5,"Spam","Not Spam"))
y_hat_ensemble<-factor(y_hat_ensemble, levels = c("Spam","Not Spam"))

confusionMatrix(y_hat_ensemble, test_set$spam)$table
sensitivity(y_hat_ensemble, test_set$spam)
specificity_df<-rbind(specificity_df,data.frame(method="ensemble", Spec=specificity(y_hat_ensemble, test_set$spam)))
specificity_df

save.image("Emails.RData")
