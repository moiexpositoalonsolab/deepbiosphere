## This script runs maxent in the R package BIOMOD for a single species
## M. Ruffley 11/10/2020
## Edits made by L Gillespie 6/6/2022
## to include RF with background sampling, and biomod ensembling

##----------------------------------------------------------------------------##
##      Helper functions for later in script                                  ##
##----------------------------------------------------------------------------##

# Convert the array + transform object to raster type
# input is a python reticulate object of type 
# returned by deepbiosphere.scripts.get_bioclim_rasters()
convert_raster <- function(input){
  arr <- input[0]
  aff <- input[1]
  # convert array to matrix, filling nan values with nan placeholder
  mat <- py_to_r(np$squeeze(arr$filled(arr$fill_value)))
  # get extent of bounds of raster from affine transform
  tl <- rasterio$transform$xy(aff, 0, 0)
  br <- rasterio$transform$xy(aff, nrow(mat), ncol(mat))
  xmn <- py_to_r(tl[0])
  xmx = py_to_r(br[0])
  ymx <- py_to_r(tl[1])
  ymn = py_to_r(br[1])
  #  make raster object, assumes that bioclim raster is EPSG:4326
  r <- raster(mat, crs=py_to_r(naip$GBIF_CRS), xmx=xmx, ymx=ymx,ymn=ymn,xmn=xmn)
  # finally, replace NAN values 
  # https://gis.stackexchange.com/questions/269314/filling-replacing-nodata-values-of-a-raster-layer-in-r
  return(reclassify(r, cbind(maxValue(r), NA)))
}

# Generates a set of 50K background points for a given species
# data: SpatialPointsDataFrame with occurrences for one species
# clim: bioclimatic raster to use as response variable.
# Note: be sure data and clim are in the same CRS before running!
# also fun fact - to call package functions from inside externally
# defined functions that are run in %dopar% you have to use the formal
# package_name::function_name syntax or else it won't work...
generate_background <- function(data, clim, nback){
  dist <- as.matrix(as.dist(raster::pointDistance(data, lonlat=TRUE)))
  diag(dist) <- NA
  # generate background climate envelope by taking circle polygons
  # around each observation where the diameter of the circle is the 
  # median distance between observations.
  Bufferpoly <- sp::polygons(dismo::circles(data, d=median(dist, na.rm=TRUE), lonlat=TRUE))
  background_clim <- raster::mask(clim, Bufferpoly)
  # allow random sampling from same grid cell as presence points 
  # following background sampling procedure from Valavi et al. 2022
  background <- dismo::randomPoints(background_clim, nback)
  background <- data.frame(background)
  names(background) <- c('longitude', 'latitude')
  return(background)
}

get_background <- function(bdir, data, clim, nback, species, dset_name, bname, poly, GBIF_CRS){
  # check if background already generated 
  dirr <- paste0(bdir, 'background/','/',dset_name,'/', bname, '/')
  fname <- paste0(dirr, species, '_background.csv') 
  if (!file.exists(fname)){
    # make background
    background <- generate_background(data, clim, nback)
    if (!(opt$band == 'unif_train_test')) {
      sp::coordinates(background) <- c('longitude', 'latitude')
      raster::projection(background) <- GBIF_CRS
      background <- background[!is.na(over(background, poly)),]
    }
    if (!dir.exists(dirr)){
      dir.create(dirr, recursive=T)
    }
    write.csv(background, fname) 
  } else {
    background <- read.csv(fname)  
    background <- as.data.frame(background)
    sp::coordinates(background) <- c('longitude', 'latitude')
    raster::projection(background) <- GBIF_CRS
  }
  return(background)  
}

run_maxent <- function(bdir, data, clim, background, sname, sdm, dset_name, bname,idCol){
  # nothreshold on best practices from Valavi et al. 2022
  maxMod <- dismo::maxent(clim, data, a=background, silent=TRUE, args = c("nothreshold"))
  # save results
  dirr <- paste0(bdir, sdm,'/', 'stats/', dset_name,'/', bname, '/')
  if (!dir.exists(dirr)){
    dir.create(dirr, recursive=T)
  }
  fname <- paste0(dirr, sname, '_maxent.csv')
  write.table(maxMod@results, file=fname, sep="\t")
  # predict to all of california
  projection <- dismo::predict(maxMod, clim) 
  # ## save raster file of probabilities of presence
  dirr <- paste0(bdir, sdm, '/', 'projections/', dset_name,'/', bname, '/')
  if (!dir.exists(dirr)){
    dir.create(dirr, recursive=T)
  }
  fname <- paste0(dirr, sname, '_maxent_Proj.tif')
  raster::writeRaster(projection, filename=fname, overwrite=TRUE)
  # extract predictions at all test set locations
  preds <- as.data.frame(list(pres_pred = raster::extract(projection, test_dset)))
  preds[[idCol]] <- test_dset[[idCol]]
  dirr <- paste0(bdir, sdm,'/', 'predictions/', dset_name,'/', bname, '/')
  if (!dir.exists(dirr)){
    dir.create(dirr, recursive=T)
  }
  fname <- paste0(dirr, sname, '_maxent_preds.csv')
  write.csv(preds, file=fname)
  return(TRUE)
}


run_rf <- function(bdir, data, clim, background, sname, sdm, dset_name, bname, test_dset, idCol){
  # convert presence + absence to a single matrix with a response column
  pres <- as.data.frame(raster::extract(clim, data))
  # 1 = presence, 0 = absence 
  pres$presence <- 1
  abs <- as.data.frame(raster::extract(clim, background))
  abs$presence <- 0
  tog <- rbind(pres, abs)
  # filter NA rows 
  dat <- na.omit(tog) 
  # get number pres, absences
  npres <- sum(dat$presence)
  if(npres == 0){
    print(paste("not enough observations for", sname, "!"))
    next
  }
  nabs <-  nrow(dat) - npres
  # convert response to factor for classification
  dat$presence <- as.factor(dat$presence)
  # make sure each bootstrapped sample as the same # of presences as absences
  samplsize <- c("0" = npres, "1" = npres)
  # presence is the value to predict: presence ~.
  # number of trees and sampling schema taken from Valavi et al. 2022
  rfMod <- randomForest::randomForest(presence ~ ., 
                                      data = dat,
                                      ntree = 1000,
                                      sampsize = samplsize,
                                      replace = TRUE)
  # save results from random forest
  conf <- as.data.frame(rfMod$confusion)
  err <- as.data.frame(rfMod$err.rate)
  votes <- as.data.frame(rfMod$votes)
  conf$datatype <- 'confusion'
  votes$datatype <- 'votes'
  err$datatype <- 'error'
  res <- plyr::rbind.fill(votes, err, conf)
  dirr <- paste0(bdir, sdm,'/', 'stats/', dset_name,'/', bname, '/')
  if (!dir.exists(dirr)){
    dir.create(dirr, recursive=T)
  }
  fname <- paste0(dirr, sname, '_rf_stats.csv')
  write.csv(res, file=fname)
  
  # now predict values on the test set
  test_data <- as.data.frame(raster::extract(clim, test_dset))
  # 1 = presence, 0 = absence 
  # don't filter NA rows, need to keep same size to transfer idCol
  preds <- as.data.frame(predict(rfMod, test_data, type='prob')) 
  names(preds) <- c('absence', 'presence')
  preds[[idCol]] <- test_dset[[idCol]]
  dirr <- paste0(bdir, sdm,'/', 'predictions/', dset_name,'/', bname, '/')
  if (!dir.exists(dirr)){
    dir.create(dirr, recursive=T)
  }
  fname <- paste0(dirr, sname, '_rf_preds.csv')
  write.csv(preds, file=fname)
  return(TRUE)
}


run_biomod <- function(data, clim, background, bdir, sdm, dset_name, bname){
  
  # change wd so projections get saved correctly
  folder <- paste0(bdir, sdm,'/' ,dset_name,'/' ,bname, '/')
  if (!dir.exists(folder)){
    dir.create(folder, recursive=T)
  }
  setwd(folder)
  # now copy over maxent jar to working directory (annoying)
  fcopy = file.copy(from = paste0(bdir, 'philips_maxent_jar/maxent.jar'),
                    to = paste0(folder))
  pres <- as.data.frame(raster::extract(clim, data))
  # 1 = presence, 0 = absence 
  pres$presence <- 1
  abs <- as.data.frame(raster::extract(clim, background))
  abs$presence <- 0
  tog <- rbind(pres, abs)
  # filter NA rows 
  dat <- na.omit(tog) 
  # PA.table is the T/F of which rows in resp.var are pseudoabsences
  PA <-  data.frame(ifelse(tog$presence == 1, FALSE, TRUE))
  # response variable must be 1 at presences and NA at pseudoabsences
  resp <-  ifelse(tog$presence == 1, 1, NA)
  bio_data <- biomod2::BIOMOD_FormatingData(resp.var = resp, 
                                            expl.var=tog[, names(tog) %in% names(clim)],
                                            resp.name=sname,
                                            PA.strategy = 'user.defined', 
                                            PA.table = PA)
  # GreenMaps used only GLM, GBM, Maxent, and RF as the models for ensembling 
  # jar <- paste(system.file(package="dismo"), "/java/maxent.jar", sep='')
  # jar <- paste0(bdir, 'philips_maxent_jar/maxent.jar')
  # maxent refuses to work, going to get script running and then return back to this
  bio_opts <- biomod2::BIOMOD_ModelingOptions( # biomod doesn't have options to tune RF, 
    # so only will tune maxent as done above
    MAXENT.Phillips = list( memory_allocated = NULL,
                            maximumiterations = 200,
                            threshold = FALSE)) # because no threshold used above for maxent
  t1 <- Sys.time()
  bio_mod <- biomod2::BIOMOD_Modeling(bio_data,
                                      models = c('GLM','GBM','RF','MAXENT.Phillips'),
                                      models.options = bio_opts,
                                      NbRunEval = 1, 
                                      DataSplit = 100, 
                                      models.eval.meth = c("ROC"), 
                                      SaveObj = FALSE,
                                      rescal.all.models = FALSE,
                                      do.full.models = TRUE,
                                      modeling.id = sname)
  t2 <- Sys.time()
  t2-t1
  bio_ens <- biomod2::BIOMOD_EnsembleModeling(modeling.output = bio_mod)
  # bio_proj <- BIOMOD_Projection(modeling.output = bio_ens,
  #                                   new.env = clim,
  #                                   proj.name = sname)
  bio_proj <- biomod2::BIOMOD_EnsembleForecasting(EM.output = bio_ens,
                                                  proj.name = "ensemble",
                                                  new.env = clim,
                                                  on_0_1000=F)
  # finally, clean up models folder, takes up too much space
  unlink(paste0(gsub('_', '.', sname),'/models'), recursive=T)
  unlink('.BIOMOD_DATA', recursive=T)
  return(TRUE)
}

##----------------------------------------------------------------------------##
## 0. Load or install packages & virtual environments
##----------------------------------------------------------------------------##

# maybe necessary since default R gcc compiler is outdated for latest version of
# RccpTOML downloading interactively in the console will allow you to update the 
# gcc version along with all dependencies but running as a script will use the 
# older version of clangand RccpTOML. Hopefully there's not any reproducibility 
# problems between the two versions... 
# install_version("RcppTOML", version = "0.1.3", repos = "http://cran.us.r-project.org")
# if you don't have any of these packages,
# install.packages('missing_package_name')

suppressPackageStartupMessages(library(biomod2))
suppressPackageStartupMessages(library(dismo))
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(rgeos))
suppressPackageStartupMessages(library(pryr))


library(usethis)
library(iterators)
library(parallel)
library(devtools)
require(devtools)
library(reticulate)
library(rJava)
library(dismo)
library(raster)
library(rgeos)
library(progress)
library(foreach)
library(randomForest)
library(doParallel)
library(sp)
library(optparse)
library(randomForest)
library(filelock)
# library(biomod2)
library(pryr)
library(plyr)
# so biomod maxent works properly
Sys.unsetenv('DISPLAY')



# Set up reticulate so it can access the virtual environment containing the
# packages for deepbiosphere. In RStudio, you only need to set use_virtualenv() 
# once and the session will save it permanently for future executions. If
# reticulate is acting up and not finding your venv, in my experience
# restarting it a few times in Rstudio will eventually get it to work¯\_ (ツ)_/¯ 
# Otherwise, try restarting with and without this line until it works
# use_virtualenv("~/deepbiosphere/dbs", required=TRUE)
# maxent.jar should come with dismo
# Sometimes dismo won't be able to find the Java jar file for maxent, although
# I've never had issues with a fresh install. There can be issues between the
# JVM version RStudio is using and the version of java maxent is expecting.
# If maxent isn't working (throws a "ClassNotFound" error) I would recommend
# just reinstalling dismo from scratch within your R session. Theoretically
# it should be smart enough to grab the right version for your R version.
# There are also many helpful threads and forums if you google `"ClassNotFound"
# dismo::maxent` that will aid in debugging
# jar <- paste(system.file(package="dismo"), "/java/maxent.jar", sep='')


# set up reticulate modules
utils <- import("deepbiosphere.Utils", convert = FALSE)
naip <- import("deepbiosphere.NAIP_Utils", convert = FALSE)
build <- import("deepbiosphere.Build_Data", convert = FALSE)
np <- import("numpy", convert = FALSE)
rasterio <- import("rasterio", convert = FALSE)
paths <- utils$paths
setwd(py_to_r(paths$BASELINES))

# set up argument parser for command line version
option_list = list(  
  make_option(c("--sdm"), type="character", default='biomod', 
              help="Which SDM to use (options are maxent (maxent), random forest (rf), and ensemble (biomod))", metavar="character"),
  make_option(c("--lonCol"), type="character", default="decimalLongitude",
                help="Name of longitude column (lat for geoclef, decimalLatitude for the big dataset)", metavar="character"),
  make_option(c("--latCol"), type="character", default="decimalLatitude",
              help="Name of latitude column (lat for geoclef, decimalLatitude for the big dataset)", metavar="character"),
  make_option(c("--dset_name"), type="character", default="big_cali_2012",
              help="Which dataset to use for training (options are big_cali_2012, geoclef_2012, big_cali_2016, goeclef_2016)", metavar="character"),
  make_option(c("--idCol"), type="character", default="gbifID",
              help="Name of column with ids of observations (either 'id' or 'gbifID'", metavar="character"),
  make_option(c("--band"), type="character", default="unif_train_test",
              help="Which band to use (either unif_train_test for interpolated set or 'band_0' - 'band_9' for extrapolates)", metavar="character"),
  make_option(c("--nback"), type="integer", default="50000", 
              help="How many background samples to use", metavar="character"),
  make_option(c("--ncpu"), type="integer", default="2",
              help="How many CPUs to use for parallel loop (should be same # of CPUs requested with SLURM)", metavar="character")
)
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

# 
# ##---------------------------------------------------------------------------##
# ## 1. Climate Data
# ##---------------------------------------------------------------------------##
# 
startime = Sys.time()

# 1. use Reticulate to grab the rasters from deepbiosphere the same way they're used for training other models (specifically using the normalize switch)
clim <- build$get_bioclim_rasters(normalized='normalize')
# generate list of R rasters, hackily from python
tostack <- list()
# can't use lapply because clim is a python object, so this is the
# cleanest way to iterate and convert to R raster type
for (i in 1:(length(clim))-1){
  tostack[[length(tostack)+1]] <- convert_raster(clim[i])
}
# stack all the rasters into one big chunk
clim <- stack(tostack)
# remove highly correlated variables
# get pairwise pearson correlation between layers
cor<-layerStats(clim,'pearson', na.rm=T)
corr_mat <- as.data.frame(cor$`pearson correlation coefficient`)
diag(corr_mat) <- NA
print("Removing correlated bioclimatic variables")
# for each layer, get what other layers have a high correlation
# 0.8 chosen based on recommendation from Valavi et al. 2022
corrd <- apply(corr_mat[-1], 1, function(x) names(which(x >0.8)))
remd <- c()
# loop through the layers removing correlated layers until
# only uncorrelated are left.
while (max(unlist(lapply(corrd, length))) > 0){
  torem <- names(which.max(lapply(corrd, length)))
  # remove each highly correlated layer
  corrd <- lapply(corrd,setdiff,torem)
  corrd[[torem]] <- NULL
  remd[[length(remd)+1]] <- torem
}
# keep only the climate layers that are low correlation
kept <- setdiff(names(corr_mat), remd)
clim <-  clim[[kept]]

##----------------------------------------------------------------------------##
## 2. Parse species data
##----------------------------------------------------------------------------##

# num. background on best practices from Valavi et al. 2022
print(paste('using sdm ', opt$sdm ,"with dataset ", opt$dset_name, " and band ", opt$band))
dset <- read.csv(paste0(py_to_r(paths$OCCS), opt$dset_name, '.csv'))

#  convert the dataset into a spatial dataset for spatial queries
sp::coordinates(dset) <- c(opt$lonCol,opt$latCol)
raster::projection(dset) <- py_to_r(naip$GBIF_CRS)

# split the dataset into test and train bands
# depending on if it's the interpolation or extrapolation experiments
if (opt$band == 'unif_train_test'){
  dset <- dset[c("species", "unif_train_test", opt$idCol)]
  train_dset <- dset[dset$unif_train_test=='train',]
  test_dset <- dset[dset$unif_train_test=='test',]
} else{
  # string_split is to remove the "band_" from the band name
  train_col <-paste0('train_', strsplit(opt$band, '_')[[1]][2])
  test_col <- paste0('test_', strsplit(opt$band, '_')[[1]][2])
  dset <- dset[c("species",  train_col, test_col, opt$idCol)]
  # because R uses a different truth variable from python...
  train_dset <- dset[dset[[paste0('train_', strsplit(opt$band, '_')[[1]][2])]] == 'True',]
  test_dset <- dset[dset[[paste0('test_', strsplit(opt$band, '_')[[1]][2])]] == 'True',]
}
# store what species are present in the train and test sets
# apparently this actually still returns just all the species
# so the code will complain some species failed, it's just 
# cause they're not actually present in that split of the 
# data, oh well. 
train_specs <- unique(train_dset$species)
test_specs <- unique(test_dset$species)
print(paste(length(train_specs), " total species in train split, ",
            length(test_specs), " total species in test split, ",
            length(intersect(train_specs, test_specs)), " species shared"))
# generate the test + train polygons for filtering occurrences and background
poly <- build$generate_split_polygons()
trains <- NA
if (opt$band != 'unif_train_test'){
  trains <- list()
  # Since reticulate can't convert between python shapely types and sp::Polygons
  # so we have to extract the coordinates and rebuild the polygon by hand.
  # first, extract the longitude, convert to numpy array, and then to R array
  for(j in 0:length(poly[[opt$band]]$train[0])){
    # only take the train polygons since that's all that's needed for spatial int.
    lon <- py_to_r(np$asarray(poly[[opt$band]]$train[j]$exterior$xy[0]))
    lat <- py_to_r(np$asarray(poly[[opt$band]]$train[j]$exterior$xy[1]))
    # create spatial polygon from boundary (complicated)
    trains[[length(trains)+1]] <- Polygons(list(Polygon(as.matrix(cbind(as.numeric(lon),as.numeric(lat))))), list(j))
  }
  # don't need to filter train dset by polygons since that was done already
  trains <- SpatialPolygons(trains)
  raster::projection(trains) <- py_to_r(naip$GBIF_CRS)
}

# make cluster w/ same # CPUs job has 
cl <- makeCluster(opt$ncpu, outfile='')
registerDoParallel(cl)

##----------------------------------------------------------------------------##
## 3. Running the SDM
##----------------------------------------------------------------------------##

# # increase the RAM for Java
options(java.parameters = "-Xmx20g" )
options(warn=-1) # suppress warning messages because maxent is very vocal
# set up progress bar to keep track of progress
pb <- progress_bar$new(
   format = "(:spin) [:bar] :percent [Elapsed time: :elapsedfull || Estimated time remaining: :eta]",
   # bit overestimated length
   total = ceiling(length(train_specs)/opt$ncpu)+50, # the way parallel processes 
   # are spawned in R means the progress bar gets copied to each parallel process
   # so it'll approximately be nspec / ncores (not perfect, since some processes)
   # will run faster so they'll do more iterations, thus the 50
   complete = "=",   # Completion bar character
   incomplete = "-", # Incomplete bar character
   current = ">",    # Current bar character
   clear = FALSE,    # If TRUE, clears the bar when finishes
   force=T)

# annoying, but reticulate doesn't work with
# parallelization since it's actually a pointer
# under the hood that can't be shared between processes
# https://stackoverflow.com/questions/69692625/parallelized-reticulate-call-with-foreach-failing
# so have to pre-convert everything I used reticulate for, before starting the loop
# sometimes maxent will throw this warning, it's okay to ignore
# https://www.ibm.com/support/pages/starting-websphere-application-server-gives-warning-message-could-not-lock-user-prefs
# actually may not be able to ignore https://community.oracle.com/tech/developers/discussion/1543444/could-not-lock-user-prefs
GBIF_CRS <- py_to_r(naip$GBIF_CRS)
bdir <- py_to_r(paths$BASELINES)
lckname <- paste0(opt$sdm,'_' , opt$band, '_', opt$dset_name, '_lockfile')
# capture and ignore the output from foreach 
toignore <- foreach (i = 1:length(train_specs), .packages = c('plyr','filelock')) %dopar% {
  
  bef <- Sys.time()
  spec <- train_specs[i]
  print(paste("starting iteration ", i, 'with species', spec))
  sname <- gsub(' ', '_', spec)
  # inexplicably, indexing below doesn't work
  # without this line when using dopar
  head(train_dset)
  data <- train_dset[train_dset$species == spec,]

  # set seed for each species for reproducibility
  set.seed(length(data) + 42)
  # get or generate background points
  background <- tryCatch(get_background(bdir, data, clim, opt$nback, sname, 
                                        opt$dset_name, opt$band, trains, GBIF_CRS),
                         error = function(e) NA)
  if (all(is.na(background))){
    print(paste("background for species ", spec, " failed!"))
    next
  }
  
  # remove background locations that are inside of the testing band
  # run maxent
  if (opt$sdm == 'maxent'){
    mod <- tryCatch(run_maxent(
          bdir, data, clim, background, sname, 
          opt$sdm, opt$dset_name, opt$band,opt$idCol),
        error = function(e) NA)
    # run random forest
  } else if(opt$sdm =='rf') {
    mod <- tryCatch(run_rf(
        bdir, data, clim, background, sname, opt$sdm, 
        opt$dset_name, opt$band, test_dset, opt$idCol),
      error = function(e) NA)
    # run biomod
  } else {
    # https://stackoverflow.com/questions/22265191/segfault-when-using-doparallel-with-rcpp-module
    library(biomod2)
    mod <- run_biomod(data, clim, background, bdir, opt$sdm, opt$dset_name, opt$band)
    # mod <- tryCatch(run_biomod(
    #     data, clim, background, bdir, opt$sdm, opt$dset_name, opt$band),
    #   error = function(e) NA)
  }
  # is this what's failing with dopar??
  if (all(is.na(mod))){
    print(paste(sdm, " for ", sname, " failed!"))
    next
  }
  aft <- Sys.time()
  # acquire lock on timing file
  lck <- lock(lckname)
  diff <- capture.output(print(aft-bef))
  pb$tick()
  timing <- as.data.frame(list(
    'species' = spec,
    'time' = diff
  ))
  folder <-paste0(bdir, opt$sdm,'/timing/' ,opt$dset_name,'/')
  if (!dir.exists(folder)){
    dir.create(folder, recursive=T)
  }
  fname <- paste0(folder, opt$band,  '_time_profiling.csv')
  # if file does not exist, write headers
  if (!file.exists(fname)){
    write.table(timing, fname, sep=',')
  } else {
    write.table(timing, fname, append=TRUE, sep=',', col.names=F)
  }
  unlock(lck)
}

stopCluster(cl)
# clean up file used for lock on progress bar
if (file.exists(lckname)){
  file.remove(lckname)
}
endtime = Sys.time()
took <- capture.output(print(endtime-startime))
print(took)
timing <- as.data.frame(list(
  'time' = took,
  'date' = Sys.time(),
  'code' = 'R',
  'model' = opt$sdm,
  'dataset' = opt$dset_name,
  'band' = opt$band,
  'cores' = opt$ncpu,
  'memory' = system('grep MemTotal /proc/meminfo', intern = TRUE)
))
fname <- paste0(paths$BASELINES, 'time_profiling.csv')
# if file does not exist, write headers
if (!file.exists(fname)){
  write.table(timing, fname, sep=',')
} else {
  write.table(timing, fname, append=TRUE, sep=',', col.names=F)
}

