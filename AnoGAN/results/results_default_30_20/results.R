library(dplyr)
library(ggplot2)

df = read.csv('score.csv')
head(df)

df %>%
  group_by(label) %>%
  summarise(mean(anomaly_score))

df %>%
  group_by(label) %>%
  summarise(min(anomaly_score))

ggplot(df, aes(x=as.factor(label), y=anomaly_score)) +
  geom_boxplot() + xlab('label')
