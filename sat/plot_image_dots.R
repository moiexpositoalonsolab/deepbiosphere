library(raster)
library(dplyr)

r<-raster("../sat/tmp/exampleExport01deg.B2.tif")
plot(r[[1]])
r

d<-read.table("../gbif/ptoy.csv",header = T)
head(d)
points(d$decimalLatitude~d$decimalLongitude)

summary(d$decimalLongitude)


d%>% 
  dplyr::filter(decimalLatitude < 45.08 & decimalLatitude > 45.0) %>%
  dplyr::filter(decimalLongitude > -115.2 & decimalLongitude < -114.7)


plot(d$decimalLatitude~d$decimalLongitude, xlim=c( -115.2,-114.7),ylim=c( 45.0,45.08))
plot(r[[1]],add=T)
