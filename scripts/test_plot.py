import numpy as np
import matplotlib as mpl

import matplotlib.pylab as pylab

fig, ax = pylab.subplots(1, 1, figsize=(6, 6))  # setup the plot

x = np.random.rand(20)  # define the data
y = np.random.rand(20)  # define the data
tag = np.random.randint(0, 20, 20)
tag[10:12] = 0  # make sure there are some 0 values to show up as grey

cmap = pylab.cm.jet  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (0.5, 0.5, 0.5, 1.0)

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "Custom cmap", cmaplist, cmap.N
)

# define the bins and normalize
bounds = np.linspace(0, 20, 21)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# make the scatter
scat = ax.scatter(
    x, y, c=tag, s=np.random.randint(100, 500, 20), cmap=cmap, norm=norm
)

# create a second axes for the colorbar
# ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
# cb = mpl.colorbar.ColorbarBase(
#     ax2,
#     cmap=cmap,
#     norm=norm,
#     # spacing="proportional",
#     ticks=bounds,
#     boundaries=bounds,
#     format="%1i",
# )
cb = pylab.colorbar(
    pylab.cm.ScalarMappable(norm=norm, cmap=cmap),
    label=r"|$V$|",
    orientation="vertical",
    ticks=bounds,
)
labels = np.arange(0, 20, 1)
loc = labels + 0.5
cb.set_ticks(loc)
cb.set_ticklabels(labels)
# ax.set_title("Well defined discrete colors")
# ax2.set_ylabel("Very custom cbar [-]", size=12)
pylab.show()
