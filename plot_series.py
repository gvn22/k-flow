from h5py import File
from matplotlib.pyplot import *

fh = File('series/series_s1.h5','r')

t = fh['scales/sim_time'][:]

ke = fh['tasks/ke'][:,0,0]
ens = fh['tasks/enstrophy'][:,0,0]

fig,ax = subplots()

plot (t,ke)
plot (t,ens)

show()