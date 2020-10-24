
library(readr)
library(tidyverse)

# source="https://archive.ics.uci.edu/ml/datasets/HCV+data"

dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00571/hcvdat0.csv", dl)

hcv<-read_csv(file = dl,na = "NA") %>% select(-X1) %>% na.omit() %>%
  mutate(Sex=factor(Sex)) %>% mutate(Category=ifelse(str_starts(Category,"0"),"Blood Donor","Hepatitis C")) %>%
  mutate(Category=factor(Category))

nrow(hcv)  

table(hcv$Category)

table(hcv$Sex)

hcv %>% ggplot(aes(Age)) + geom_histogram(binwidth = 5)

