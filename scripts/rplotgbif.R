library(dplyr)
library(moiR)
library(data.table)

# d = read.delim("gbif/toy.csv",sep="\t",header = T)
d = fread("gbif/toy.csv",header = T)
d$decimalLatitude %>% summary

d$decimalLatitude[which(is.na(d$decimalLatitude))]

dim(d)

plot(d$decimalLatitude ~d$decimalLongitude)
colnames(d) %>%
  as.matrix

mycolumns<-c("decimalLatitude","decimalLongitude","family")
which(colnames(d) %in% mycolumns)

d$speciesKey
d$order %>% fc %>% unique
d$taxonRank %>% fc %>% unique

countfam<-table(d$order %>% fc)
