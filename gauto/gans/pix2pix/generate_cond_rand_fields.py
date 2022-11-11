"""
Create conditional random fields from cpts

"""
from BroReader import read_BRO
import random
import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
from simplification.cutil import (
    simplify_coords
)
import plotly.express as px
from itertools import zip_longest
import csv


def read_bro(location):
    # read cpts from the bro around a location
    radius_distance = 0.04
    c = read_BRO.read_cpts(location, radius_distance)
    return c

def create_random_fields(
    std_value, 
    mean, 
    v_min, 
    v_max, 
    aniso_x, 
    aniso_z, 
    ndim, 
    lognormal, 
    seed, 
    cond_pos, 
    cond_val,
    theta
):
    len_scale = np.array([aniso_x, aniso_z]) * theta
    if lognormal:
        var = np.log((std_value / mean) ** 2 + 1)
        mean = np.log(mean**2 / (np.sqrt(mean**2 + std_value**2)))
    else:
        var = std_value**2

    model = gs.Exponential(dim=ndim, var=var, len_scale=len_scale)
    krige = gs.Krige(model, cond_pos=cond_pos, cond_val=cond_val, mean=0)
    cond_srf = gs.CondSRF(krige, seed=seed)
    return cond_srf

def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    
    if window == 'flat':  # moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]



if __name__ == "__main__":
    # read cpts
    location = [90061, 463105]
    cpts = read_bro(location)
    # keep 70% of the data
    RF_data = random.choices(cpts, k=int(len(cpts) * 1))
    for counter, cpt in enumerate(RF_data):
        RF_data[counter]['IC'] = smooth(cpt['IC'], window_len=40)
    # set values 
    std_value = 0.3
    mean = 2.0
    v_min = 1
    v_max = 3
    theta = 2
    aniso_x = 6
    aniso_z = 1
    ndim = 2
    lognormal = True
    seed = 14
    ens_no = 4
    #seed = gs.random.MasterRNG(seed)
    for pair_cpts in zip(RF_data, RF_data[1:]):
        if not(pair_cpts[0]['name'] == pair_cpts[1]['name']):
            print(f"Creating rf for cpts{pair_cpts[0]['name']} and {pair_cpts[1]['name']}")
            y, z, cond_val = [], [], []
            for data in pair_cpts:

                # simplify
                fake_coords = np.array([list(data['depth']), list(data['IC'])]).T
                simplified = simplify_coords(fake_coords, 0.01).T
                y += len(list(simplified[0]))*[abs(location[1] - data['coordinates'][1])]            
                z += list(simplified[0])
                cond_val += list(simplified[1])
            cond_pos = [y, z]
            theta
            srf = create_random_fields(
                std_value, mean, v_min, v_max, aniso_x, aniso_z, ndim, lognormal, seed, cond_pos, cond_val, theta
            )
            y_grid = np.linspace(min(y), max(y), 64)
            z_grid = np.linspace(min(z), max(z), 64)
            ys, zs = np.meshgrid(y_grid, z_grid, indexing="ij")
            srf.set_pos([ys, zs], "unstructured")

            for i in range(ens_no):
                srf(seed=seed + i, store=[f"fld{i}", False, False])


                test_input = np.reshape(srf[i], (64,64))
                fig = px.imshow(test_input.T)
                fig.write_html(f"data\cond_rf\\rf_cpts{pair_cpts[0]['name']}_{pair_cpts[1]['name']}_{i}.html")

                d = [ ys.flatten(), zs.flatten(), srf[i].flatten()]

                export_data = zip_longest(*d, fillvalue="")
                with open(
                    f"data\\cond_rf\\rf_cpts{pair_cpts[0]['name']}_{pair_cpts[1]['name']}_{i}.csv", "w", encoding="ISO-8859-1", newline=""
                ) as myfile:
                    wr = csv.writer(myfile)
                    wr.writerow(("x", "z", "IC"))
                    wr.writerows(export_data)
                myfile.close()

            ens_no_plot = 4
            fig, ax = plt.subplots(ens_no_plot + 1, ens_no_plot + 1, figsize=(8, 8))
            # plotting kwargs for scatter and image
            vmax = np.max(srf.all_fields)
            sc_kw = dict(c=cond_val, edgecolors="k", vmin=0, vmax=vmax)
            im_kw = dict(vmin=0, vmax=vmax)
            for i in range(ens_no_plot):
                # conditioned fields and conditions
                ax[i + 1, 0].contourf(ys.T, zs.T, np.reshape(srf[i], (64,64)).T)
                ax[i + 1, 0].invert_yaxis()
                ax[i + 1, 0].scatter(cond_pos[0],cond_pos[1], **sc_kw)
                ax[i + 1, 0].set_ylabel(f"Field {i}", fontsize=10)
                ax[0, i + 1].contourf(ys.T, zs.T, np.reshape(srf[i], (64,64)).T)
                ax[0, i + 1].invert_yaxis()
                ax[0, i + 1].scatter(*cond_pos, **sc_kw)
                ax[0, i + 1].set_title(f"Field {i}", fontsize=10)
                # absolute differences
                for j in range(ens_no_plot):
                    ax[i + 1, j + 1].imshow(np.abs(np.reshape(srf[i], (64,64)) - np.reshape(srf[j], (64,64))).T, **im_kw)

            # beautify plots
            ax[0, 0].axis("off")
            for a in ax.flatten():
                a.set_xticklabels([]), a.set_yticklabels([])
                a.set_xticks([]), a.set_yticks([])
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.savefig(f"data\\cond_rf\\rf_cpts{pair_cpts[0]['name']}_{pair_cpts[1]['name']}.png")