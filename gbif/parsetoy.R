library(dplyr)
library(moiR)
library(ggplot2)
library(cowplot)

d = read.delim("toy.csv",sep="\t",header = T)

#  quick plot
plot(d$decimalLatitude ~d$decimalLongitude)


# columns of interest
colnames(d) %>%
  as.matrix

mycolumns<-c("decimalLatitude","decimalLongitude","family")
which(colnames(d) %in% mycolumns)

# intersting coordinates
d$coordinatePrecision %>% summary
d$coordinateUncertaintyInMeters %>% summary

# get top families
countmy<-table(d$order %>% fc)
mytops<-names(countmy)[which(fn(countmy)>10)]

d$family<-fc(d$family)
d$order<-fc(d$order)
d %>%
  dplyr::select(decimalLatitude,decimalLongitude, order) %>%
  dplyr::mutate(order=fn(order=="Caryophyllales"))%>%
  dplyr::filter(decimalLatitude<50 & 
                  decimalLongitude< -50 & 
                  decimalLongitude > -140) ->
  sd
dim(sd)
head(sd)
write.table(sd,"../gbif/ptoy.csv",col.names = T,row.names = F,quote = F,sep="\t")
qplot(y=sd$decimalLatitude , sd$decimalLongitude, color=sd$order)
head(sd)

sd %>%
  dplyr::filter(order %in% c(mytops)) ->
ssd
dim(ssd)


# write.tsv(ssd,"stoy.csv")

pdf("mapobservations.pdf",height = 5,width = 16)
ggplot(sd,aes(x=decimalLongitude,y=decimalLatitude,color=order)) +
  geom_point()
dev.off()

dim(sd)
unique(d$species)
