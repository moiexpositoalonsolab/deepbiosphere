## To work with Earth Engine
import ee
import ee.mapclient
ee.Initialize()
## To download
import urllib.request
## To read images
import geopandas as gpd
import rasterio
from rasterio import features


################################################################################
## Utilities
def readraster(file="tmp/SRTM90_V4.elevation.tif")
    rst= rasterio.open(file)
    return(rst)

def readimgnp(file='../sat/exampleExport01deg.B1.tif')
    im=:wq

#################################################################################
### Interact with Google Earth Engine
#### List of available resources
#
#
#### Extract pixel values per location
## from https://stackoverflow.com/questions/46980815/google-earthengine-time-series-of-reduceregion
#feature_geometry = {
#    'type': 'MultiPolygon',
#    'coordinates': [[[
#        [-120, 35],
#        [-120.001, 35],
#        [-120.001, 35.001],
#        [-120, 35.001],
#        [-120, 35]
#    ]]]
#}
#
#collection = ee.ImageCollection(
#    'MODIS/006/MOD13Q1').filterDate('2017-01-01', '2017-05-01')
#
#def setProperty(image):
#    dict = image.reduceRegion(ee.Reducer.mean(), feature_geometry)
#    return image.set(dict)
#
#withMean = collection.map(setProperty)
#
#print (withMean.aggregate_array('NDVI').getInfo())
#########
#import ee
#import ee.mapclient
#import urllib.request
#ee.Initialize()
#
## # // Load Landsat 8 top-of-atmosphere (TOA) input imagery.
## # var collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA').select('B[1-7]');
##
## collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA').select('B[1-7]');
##
## # # // Define a region of interest as a buffer around a point.
## # var geom = ee.Geometry.Point(-122.08384, 37.42503).buffer(500);
## geom = ee.Geometry.Point(-122.08384, 37.42503).buffer(500);
##
## print(geom)
##
## print(ui.Chart.image.series(collection, geom, ee.Reducer.mean(), 30));
#
## // Load a landsat image and select three bands.
## landsat = ee.Image('LANDSAT/LC08/C01/T1_TOA/LC08_123032_20140515').select(['B4', 'B3', 'B2']);
##
## # // Create a geometry representing an export region.
## geometry = ee.Geometry.Rectangle([116.2621, 39.8412, 116.4849, 40.01236]);
##
## # // Export the image, specifying scale and region.
## ee.batch.Export.image.toDrive(
##   image= landsat,
##   description= 'imageToDriveExample',
##   scale= 30 #,
##   # region= geometry
## );
#
## ## from https://stackoverflow.com/questions/41989977/pixel-values-google-earth-engine
## # // Image
## # var im1 = ee.Image('COPERNICUS/S2/20160422T084804_20160422T123809_T36TVK')
## im1 = ee.Image('COPERNICUS/S2/20160422T084804_20160422T123809_T36TVK')
##
## # // Point
## # var p = ee.Geometry.Point(32.3, 40.3)
## p = ee.Geometry.Point(32.3, 40.3)
##
## # // Extract the data
## # var data = im1
## # .select("B3")
## # .reduceRegion(ee.Reducer.first(),p,10)
## # .get("B3")
## data = im1.select("B3").reduceRegion(ee.Reducer.first(),p,10).get("B3")
##
##
## # // Convert to Number for further use
## # var dataN = ee.Number(data)
## dataN = ee.Number(data).get("B3")
##
## # // Show data
## # print(dataN)
## print(dataN)
## print(data)
##
## # // Add Layers
## # Map.centerObject(im1)
## # Map.addLayer(im1,{bands:["B4","B3","B2"],min:0,max:5000})
## # Map.addLayer(p)
#
#
## from https://stackoverflow.com/questions/46980815/google-earthengine-time-series-of-reduceregion
#import ee
#ee.Initialize()
#
#feature_geometry = {
#    'type': 'MultiPolygon',
#    'coordinates': [[[
#        [-120, 35],
#        [-120.001, 35],
#        [-120.001, 35.001],
#        [-120, 35.001],
#        [-120, 35]
#    ]]]
#}
#
#collection = ee.ImageCollection(
#    'MODIS/006/MOD13Q1').filterDate('2017-01-01', '2017-05-01')
#
#def setProperty(image):
#    dict = image.reduceRegion(ee.Reducer.mean(), feature_geometry)
#    return image.set(dict)
#
#withMean = collection.map(setProperty)
#
#print (withMean.aggregate_array('NDVI').getInfo())
#########
## from https://geoscripting-wur.github.io/Earth_Engine/
#import ee
#from ee import batch
### Initialize connection to server
#ee.Initialize()
### Define your image collection 
#collection = ee.ImageCollection('LANDSAT/LC8_L1T_TOA')
### Define time range
#collection_time = collection.filterDate('2013-04-11', '2018-01-01') #YYYY-MM-DD
### Select location based on location of tile
#path = collection_time.filter(ee.Filter.eq('WRS_PATH', 198))
#pathrow = path.filter(ee.Filter.eq('WRS_ROW', 24))
## or via geographical location:
##point_geom = ee.Geometry.Point(5, 52) #longitude, latitude
##pathrow = collection_time.filterBounds(point_geom)
### Select imagery with less then 5% of image covered by clouds
#clouds = pathrow.filter(ee.Filter.lt('CLOUD_COVER', 5))
### Select bands
#bands = clouds.select(['B4', 'B3', 'B2'])
### Make 8 bit data
#def convertBit(image):
#    return image.multiply(512).uint8()  
### Convert bands to output video  
#outputVideo = bands.map(convertBit)
#print("Starting to create a video")
### Export video to Google Drive
#out = batch.Export.video.toDrive(outputVideo, description='Netherlands_video_region_L8_time', dimensions = 720, framesPerSecond = 2, region=([5.588144,51.993435], [5.727906, 51.993435],[5.727906, 51.944356],[5.588144, 51.944356]), maxFrames=10000)
### Process the image
#process = batch.Task.start(out)
#print("Process sent to cloud")
