library(dplyr)
library(moiR)

d = read.delim("toy.csv",sep="\t",header = T)

plot(d$decimalLatitude ~d$decimalLongitude)
colnames(d) %>%
  as.matrix

mycolumns<-c("decimalLatitude","decimalLongitude","family")
which(colnames(d) %in% mycolumns)

d$speciesKey
d$order %>% fc %>% unique
d$taxonRank %>% fc %>% unique

countfam<-table(d$order %>% fc)
which(>10)
