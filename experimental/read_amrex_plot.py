import os
import yt
from yt.frontends.boxlib.data_structures import AMReXDataset
import matplotlib.pyplot as plt

data_folder = "/pscratch/sd/a/ajnonaka/rtil/data"
plot_file = "plt0004600"

if __name__=='__main__':
    plot_list = os.listdir(data_folder)
    plot_list.sort()

    plot_path = os.path.join(data_folder, plot_file)
    print(plot_path)

    ds = AMReXDataset(plot_path)
    ad0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    charge_numpy_array = ad0['charge'].to_ndarray()

    sl = charge_numpy_array[128, :, :]

    # Plot sl and save to a png file
    fig, ax = plt.subplots()
    ax.imshow(sl)
    fig.savefig('../outputs/charge.png')
    

