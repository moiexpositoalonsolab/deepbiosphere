"""
Parsing and gridding GBIF [taxon, latitude, longitude] .csv file
@author: moisesexpositoalonso@gmail.com

"""

import pandas as pd
import os
import numpy as np


###########################################################################################
def subcoor(d,lat,lon):
    d_ = d.loc[d.iloc[:,1]<lat+1].loc[d.iloc[:,1]>lat].loc[d.iloc[:,2]<lon+1].loc[d.iloc[:,2]>lon]
    return(d_)

def makegrid(n):
    a=np.array(list(range(n)))+1 # axis with offset for 0 base index to 1
    points=product(a,repeat=2) #only allow repeats for (i,j), (j,i) pairs with i!=j
    return(np.asarray(list(points)) )

def maketensor(x,y,z):
    a = np.zeros((x, y, z))
    return(a)

def makespphash(iterable):
    seen = set()
    result = []
    for element in iterable:
        hashed = element
        if isinstance(element, dict):
            hashed = tuple(sorted(element.iteritems()))
        elif isinstance(element, list):
            hashed = tuple(element)
        if hashed not in seen:
            result.append(element)
            seen.add(hashed)
    return result

################################################################################

def readgbif(path="../gbif/pgbif.csv",sep="\t"):
    d = pd.read_csv(path, sep)
    #print('Load GBIF file with #rows = %i' %(d.size))
    return(d)

def tensorgbif():
    spp=makespphash(d.iloc[:,0])
    spptot=len(spp)
    sppdic=make_sppdic(spp,spptot)
    #tens=maketensor(10,10,spptot) # this for future implementation
    return('notimplemented yet')


def whichwindow(w,v):
    count=0
    for i in w:
        if v>=i[0] and v<i[1]:
            break
        else:
            count=count+1
    return count

def iffamily(val,fam):
    if val==fam:
        return(1)
    else:
        return(0)

def tensoronetaxon(step, breaks, lat,lon, d, sppname):
    # Subset to SF
    d_ = subcoor(d,lat,lon)
    cactusnum=int(d_[d_['family'] == sppname].size)
    # print('There are {} {} within this grid'.format(sppname,cactusnum))
    ## make grid steps
    sb= step/breaks
    xwind=[[lon+(sb*i),lon+(sb*(i+1))]  for i in range(int(breaks))]
    ywind=[[lat+(sb*i),lat+(sb*(i+1))]  for i in range(int(breaks))]
    ####################################################
    # Fill tensor
    tens=maketensor(2,breaks+1,breaks+1)#only for cactaceae
    for index, r in d_.iterrows():
        # print(r)
        da=whichwindow(ywind,r[1])
        do=whichwindow(xwind,r[2])
        dspp=iffamily(r[0],sppname)
        tens[dspp,da,do]= tens[dspp,da,do] +1
    # total observation per grid
    totobs=tens.sum(axis=0)
    # % of cactaceae
    cactae=tens[1,:,:]
    # cactae=tens[1,:,:]/(totobs+0.0001)
    # cactae=(cactae>0.0001)*1
    cactae=(cactae>0)*1
    return(cactae)







# def make_sppdic(spp,total):
#     sppdic={}
#     for i in range(0,total):
#         sppdic[i]=spp[i]
#     return(sppdic)
#
# def make_cacdic(spp,total):
#     sppdic={'Cactaceae':1 , 'NoCactaceae':0}
#     return(sppdic)

# def make_locdic(lon,totalwindows=1,windowstep=0.1):
#     locdic={}
#     lon=round(lon,1)
#     for i in range(0,totalwindows):
#         locdic[i]=round(lon,1)
#         lon=lon+windowstep
#     return(locdic)

def key_for_value(d, value):
    # this will be useful for final implementation
    return(list(d.keys())[list(d.values()).index(value)])

# # generate translators of location
# londic=make_locdic(lon,breaks+1)
# latdic=make_locdic(lat,breaks+1)
# # total cactus
# # d_[d_['family']=='Cactaceae'].size
# d_[d_['family']=='Brassicaceae'].size
#
