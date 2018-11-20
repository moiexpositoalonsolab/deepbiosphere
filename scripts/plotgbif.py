import pandas as pd
import os
import matplotlib.pyplot as plt

# fix location
#cwd = os.getcwd()
#print(cwd)
#os.chdir("~/moisesexpositoalonso/deepbios/dbio")

# read toy dataset
#d = pd.read_csv("ptoy.csv",sep="\t",header=0)
d = pd.read_csv("../gbif/0002455-181108115102211.csv")
d.head()


#################################################################################
# plot
#d.plot.scatter(x='decimalLongitude', y='decimalLatitude')
#y=d.iloc[:,0].values 
#x=d.iloc[:,1].values    
#
#print(y)
#print(x)
#
#f = plt.figure()
#plt.plot(x,y, "o")
#plt.show()
#f.savefig("foo.pdf", bbox_inches='tight')

