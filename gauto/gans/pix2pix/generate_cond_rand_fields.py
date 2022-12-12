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
import pyproj
import folium
from scipy.spatial import KDTree
import time


def read_bro(location):
    # read cpts from the bro around a location
    radius_distance = 0.4
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
    len_scale = np.array([aniso_x, aniso_x + 1, aniso_z]) * theta
    if lognormal:
        var = np.log((std_value / mean) ** 2 + 1)
        mean = np.log(mean**2 / (np.sqrt(mean**2 + std_value**2)))
    else:
        var = std_value**2

    model = gs.Exponential(dim=ndim, var=var, len_scale=len_scale, angles=np.pi * 3)
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

def create_validation_dataset(cpt_data, location):
    cpt_point_collection = [data['coordinates'] for data in cpt_data]
    # simplify all cpts
    for data in cpt_data:
        fake_coords = np.array([list(data['depth']), list(data['IC'])]).T
        simplified = simplify_coords(fake_coords, 0.1).T
        data['depth'] = simplified[0]
        data['IC'] = simplified[1]
    # create closest pairs of threes
    tree = KDTree(cpt_point_collection)
    nearest_dist, nearest_ind = tree.query(cpt_point_collection, k=4)
    for counter, ind in enumerate(nearest_ind):
        x, y, z, cond_val = [], [], [], []    
        # reshape to the 64 step values
        for sub_ind in ind:
            data = cpt_data[sub_ind]
            x += len(list(data['depth'])) *[abs(location[0] - data['coordinates'][0])] 
            y += len(list(data['depth'])) *[abs(location[1] - data['coordinates'][1])]            
            z += list(data['depth'])
            cond_val += list(data['IC'])
        x_grid = np.linspace(min(x), max(x), 64)
        y_grid = np.linspace(min(y), max(y), 64)
        z_grid = np.linspace(min(z), max(z), 64)
        distinct_y = [abs(location[1] - cpt_data[sub_ind]['coordinates'][1]) for sub_ind in ind]
        input_idx = []
        values_result = np.zeros((64, 64))
        for counter_2, y in enumerate(distinct_y):
            values = abs(y_grid - y)
            input_idx = list(values).index(min(values))
            values_result[input_idx] = np.interp(z_grid, cpt_data[ind[counter_2]]['depth'], cpt_data[ind[counter_2]]['IC'])
        
        plt.imshow(values_result.T)
        plt.savefig(f"data\\cond_rf\\validation_final\\validation{counter}.png")
        plt.clf()

        ys, zs = np.meshgrid(y_grid, z_grid, indexing="ij")
        d = [ ys.flatten(), zs.flatten(), values_result.flatten()]
        export_data = zip_longest(*d, fillvalue="")
        with open(
                f"data\\cond_rf\\test\\rf_cpts_{counter}.csv", "w", encoding="ISO-8859-1", newline=""
        ) as myfile:
            wr = csv.writer(myfile)
            wr.writerow(("x", "z", "IC"))
            wr.writerows(export_data)
        myfile.close()


if __name__ == "__main__":
    # read cpts
    location = [145704.420, 442456.710 ]
    #location = [144344, 442344]
    cpts = read_bro(location)
    # keep 70% of the data
    RF_data = [data for data in cpts if len(data['IC']) > 100]
    print(f"cpt number {len(RF_data)}")
    for counter, cpt in enumerate(RF_data):
        RF_data[counter]['IC'] = smooth(cpt['IC'], window_len=40)
    # set values 
    std_value = 0.3
    mean = 2.0
    v_min = 1
    v_max = 3
    theta = 7
    aniso_x = 20
    aniso_z = 1
    ndim = 3
    lognormal = True
    seed = 14
    ens_no = 2
    number_of_RFs = 2
    percentage_data = 0.8
    TEST = False

    if TEST:
        create_validation_dataset(RF_data, location=location)
    
    for rf_id in range(number_of_RFs):
        subset_data = random.choices(cpts, k=5)
        coordinates_list = np.array([[cpt['coordinates'][0], cpt['coordinates'][1]]for cpt in subset_data])
        m = folium.Map(control_scale=True)
        tile = folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        # plot array
        transformer = pyproj.Transformer.from_crs(28992, 4326, always_xy=True)
        for point in coordinates_list:
            lon, lat = transformer.transform(point[0], point[1])
            marker = folium.Marker(location=[lat, lon])
            marker.add_to(m)
        m.save(f"data\\cond_rf\\test\\map_cpts_{rf_id}.html")
        
        x, y, z, cond_val = [], [], [], []
        for data in subset_data:
            # simplify
            fake_coords = np.array([list(data['depth']), list(data['IC'])]).T
            simplified = simplify_coords(fake_coords, 0.1).T
            x += len(list(simplified[0]))*[abs(location[0] - data['coordinates'][0])]    
            y += len(list(simplified[0]))*[abs(location[1] - data['coordinates'][1])]            
            z += list(simplified[0])
            cond_val += list(simplified[1])
        cond_pos = [x, y, z]
        print("Create cond RF model...")
        srf = create_random_fields(
            std_value, mean, v_min, v_max, aniso_x, aniso_z, ndim, lognormal, seed, cond_pos, cond_val, theta
        )
        print("Created cond RF model...")
        x_grid = np.linspace(min(x), max(x), 64)
        y_grid = np.linspace(min(y), max(y), 64)
        z_grid = np.linspace(min(z), max(z), 64)
        xs, ys, zs = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
        srf.set_pos([xs, ys, zs], "unstructured")
        print("Creating cond RF ...")
        for i in range(ens_no):
            start = time.time()
            srf(seed=seed + i, store=[f"fld{i}", False, False])
            end = time.time()
            print(f"Time it took to create RF {end - start}")
            test_input = np.reshape(srf[i], (64,64,64))
            test_input = np.array([input_i.T for input_i in test_input])
            fig = px.imshow(test_input, 
                            animation_frame=0, 
                            labels=dict(animation_frame="slice"),
                            zmin=0,
                            zmax=4)
            fig.write_html(f"data\cond_rf\\test\\rf_cpts{rf_id}_{i}.html")

            d = [xs.flatten(),  ys.flatten(), zs.flatten(), srf[i].flatten()]

            export_data = zip_longest(*d, fillvalue="")
            with open(
                    f"data\\cond_rf\\test\\rf_cpts{rf_id}_{i}.csv", "w", encoding="ISO-8859-1", newline=""
            ) as myfile:
                wr = csv.writer(myfile)
                wr.writerow(("x", "y", "z", "IC"))
                wr.writerows(export_data)
            myfile.close()
            print("Created cond RF ...")

        #ens_no_plot = 4
        #fig, ax = plt.subplots(ens_no_plot + 1, ens_no_plot + 1, figsize=(8, 8))
        ## plotting kwargs for scatter and image
        #vmax = np.max(srf.all_fields)
        #sc_kw = dict(c=cond_val, edgecolors="k", vmin=0, vmax=vmax)
        #im_kw = dict(vmin=0, vmax=vmax)
        #for i in range(ens_no_plot):
        #    # conditioned fields and conditions
        #    ax[i + 1, 0].contourf(ys.T, zs.T, np.reshape(srf[i], (64,64)).T)
        #    ax[i + 1, 0].invert_yaxis()
        #    ax[i + 1, 0].scatter(cond_pos[0],cond_pos[1], **sc_kw)
        #    ax[i + 1, 0].set_ylabel(f"Field {i}", fontsize=10)
        #    ax[0, i + 1].contourf(ys.T, zs.T, np.reshape(srf[i], (64,64)).T)
        #    ax[0, i + 1].invert_yaxis()
        #    ax[0, i + 1].scatter(*cond_pos, **sc_kw)
        #    ax[0, i + 1].set_title(f"Field {i}", fontsize=10)
        #    # absolute differences
        #    for j in range(ens_no_plot):
        #        ax[i + 1, j + 1].imshow(np.abs(np.reshape(srf[i], (64,64)) - np.reshape(srf[j], (64,64))).T, **im_kw)
        ## beautify plots
        #ax[0, 0].axis("off")
        #for a in ax.flatten():
        #    a.set_xticklabels([]), a.set_yticklabels([])
        #    a.set_xticks([]), a.set_yticks([])
        #fig.subplots_adjust(wspace=0, hspace=0)
        #fig.savefig(f"data\\cond_rf\\rf_cpts{rf_id}.png")
        #plt.clf()