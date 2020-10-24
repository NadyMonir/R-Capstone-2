
library(readr)
library(tidyverse)
library(caret)

# source="https://archive.ics.uci.edu/ml/datasets/HCV+data"

dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00571/hcvdat0.csv", dl)

hcv<-read_csv(file = dl,na = "NA") %>% select(-X1) %>% na.omit() %>%
  mutate(Sex=factor(Sex)) %>% mutate(Category=ifelse(str_starts(Category,"0"),"Blood Donor","Hepatitis C")) %>%
  mutate(Category=factor(Category, levels = c("Hepatitis C","Blood Donor")))

nrow(hcv)  

table(hcv$Category)

table(hcv$Sex)

table(hcv$Sex,hcv$Category)

hcv %>% ggplot(aes(Age)) + geom_histogram(binwidth = 5)

hcv %>% mutate(Age=round(Age/5)*5) %>% group_by(Age) %>% summarise(mean=mean(Category=="Hepatitis C")) %>%
  ggplot(aes(Age,mean)) +geom_line()

plot_test_Category<-hcv %>% select(-Sex, -Age) %>% 
  pivot_longer(names_to = "blood_test", values_to="test_result",cols=-Category) %>%
  ggplot(aes(Category,test_result, fill=Category))+
  geom_boxplot()+
  facet_wrap(~blood_test, scales = "free")+
  theme(axis.text.x = element_blank())
plot_test_Category

############

test_index<-createDataPartition(hcv$Category, times = 1, p = 0.2, list = F)
train_set<-hcv[-test_index,]
test_set<-hcv[test_index,]
nrow(train_set)

############

modelLookup("glm")

control <- trainControl(summaryFunction = twoClassSummary) 

train_glm<-train(Category~., method="glm", data=train_set, metric = "Sens", trControl=control )
y_hat_glm<-predict(train_glm, newdata = test_set)
confusionMatrix(y_hat_glm, test_set$Category)$table
sensitivity(y_hat_glm, test_set$Category)
sensitivity_df<-data.frame(method="glm", Sens=sensitivity(y_hat_glm, test_set$Category))

############

modelLookup("lda")

train_lda<-train(Category~., method="lda", data=train_set, metric = "Sens", trControl=control )
y_hat_lda<-predict(train_lda, newdata = test_set)
confusionMatrix(y_hat_lda, test_set$Category)$table
sensitivity(y_hat_lda, test_set$Category)
sensitivity_df<-rbind(sensitivity_df,data.frame(method="lda", Sens=sensitivity(y_hat_lda, test_set$Category)))

############

modelLookup("qda")


train_qda<-train(Category~., method="qda", data=train_set, metric = "Sens", trControl=control )
y_hat_qda<-predict(train_qda, newdata = test_set)
confusionMatrix(y_hat_qda, test_set$Category)$table
sensitivity(y_hat_qda, test_set$Category)
sensitivity_df<-rbind(sensitivity_df,data.frame(method="qda", Sens=sensitivity(y_hat_qda, test_set$Category)))
sensitivity_df


############

modelLookup("knn")

grid <- data.frame(k = seq(2, 30))

train_knn<-train(Category~., method="knn", data=train_set, metric = "Sens", trControl=control, tuneGrid=grid )
y_hat_knn<-predict(train_knn, newdata = test_set)
confusionMatrix(y_hat_knn, test_set$Category)$table
sensitivity(y_hat_knn, test_set$Category)
sensitivity_df<-rbind(sensitivity_df,data.frame(method="knn", Sens=sensitivity(y_hat_knn, test_set$Category)))
sensitivity_df

############

modelLookup("gamLoess")

grid <- expand.grid(span = seq(0.15, 0.65, len = 10), degree = 1)

train_loess<-train(Category~., method="gamLoess", data=train_set, metric = "Sens", trControl=control , tuneGrid=grid)
ggplot(train_loess, highlight = T)

y_hat_loess<-predict(train_loess, newdata = test_set)
confusionMatrix(y_hat_loess, test_set$Category)$table
sensitivity(y_hat_loess, test_set$Category)
sensitivity_df<-rbind(sensitivity_df,data.frame(method="gamLoess", Sens=sensitivity(y_hat_loess, test_set$Category)))
sensitivity_df


############

modelLookup("rpart")

grid <- data.frame(cp=seq(0,0.2, length.out = 21))

train_rpart<-train(Category~., method="rpart", data=train_set, metric = "Sens", trControl=control , tuneGrid=grid)
ggplot(train_rpart, highlight = T)

plot(train_rpart$finalModel,margin = 0.1)
text(train_rpart$finalModel, cex=0.75)

y_hat_rpart<-predict(train_rpart, newdata = test_set)
confusionMatrix(y_hat_rpart, test_set$Category)$table
sensitivity(y_hat_rpart, test_set$Category)
sensitivity_df<-rbind(sensitivity_df,data.frame(method="rpart", Sens=sensitivity(y_hat_rpart, test_set$Category)))
sensitivity_df


############

modelLookup("rf")

grid <- data.frame(mtry=seq(2,12))

train_rf<-train(Category~., method="rf", data=train_set, metric = "Sens", trControl=control , tuneGrid=grid)

ggplot(train_rf, highlight = T)

y_hat_rf<-predict(train_rf, newdata = test_set)
confusionMatrix(y_hat_rf, test_set$Category)$table
sensitivity(y_hat_rf, test_set$Category)
sensitivity_df<-rbind(sensitivity_df,data.frame(method="rf", Sens=sensitivity(y_hat_rf, test_set$Category)))
sensitivity_df

#############

ensemble<-data.frame(y_hat_glm,y_hat_lda,y_hat_qda, y_hat_loess, y_hat_knn,y_hat_rpart,y_hat_rf)
y_hat_ensemble <-apply(ensemble,1 , function(x) ifelse(mean(x=="Hepatitis C")>0.5,"Hepatitis C","Blood Donor"))
y_hat_ensemble<-factor(y_hat_ensemble, levels = c("Hepatitis C","Blood Donor"))

confusionMatrix(y_hat_ensemble, test_set$Category)$table
sensitivity(y_hat_ensemble, test_set$Category)
sensitivity_df<-rbind(sensitivity_df,data.frame(method="ensemble", Sens=sensitivity(y_hat_ensemble, test_set$Category)))
sensitivity_df

# train_lda=train(Category~., method="gamLoess", data = hcv, preProcess="center")
# plot(train_glm$finalModel,margin = 0.1)
#  text(train_glm$finalModel, cex=0.75)
#  ggplot(train_glm, highlight=T)
# varImp(train_lda)
 # grid<-expand.grid(span=seq(0.1,2,0.1), degree=1:2)