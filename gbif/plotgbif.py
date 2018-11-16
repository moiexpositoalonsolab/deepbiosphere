import pandas as pd
import os
import matplotlib.pyplot as plt

# fix location
#cwd = os.getcwd()
#print(cwd)
#os.chdir("~/moisesexpositoalonso/deepbios/dbio")

# read toy dataset
d = pd.read_csv("ptoy.csv",sep="\t",header=0)
print(d.head())

# plot
#d.plot.scatter(x='decimalLongitude', y='decimalLatitude')
#ax.show()

#d[["decimalLatitude"]].head()


y=d.iloc[:,0].values
x=d.iloc[:,1].values    

print(y)
print(x)
#x.head()
#y.head()


f = plt.figure()
plt.plot(x,y, "o")
plt.show()
f.savefig("foo.pdf", bbox_inches='tight')




#print(d['decimalLongitude'].dtypes)

#d[[1]].head()

#ax.scatter(


# example plotting
#from numpy.random import rand
#
#
#fig, ax = plt.subplots()
#for color in ['red', 'green', 'blue']:
#    n = 750
#    x, y = rand(2, n)
#    scale = 200.0 * rand(n)
#    ax.scatter(x, y, c=color, s=scale, label=color,
#               alpha=0.3, edgecolors='none')
#
#ax.legend()
#ax.grid(True)
#
#plt.show()
