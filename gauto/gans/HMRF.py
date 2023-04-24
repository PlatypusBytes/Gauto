import numpy as np
from hmmlearn import hmm
import pandas as pd
import matplotlib.pyplot as plt

from BroReader import read_BRO

def read_bro(location):
    # read cpts from the bro around a location
    radius_distance = 0.4
    c = read_BRO.read_cpts(location, radius_distance)
    return c

if __name__ == '__main__':
    # read cpts from the bro around a location
    location = [145704.420, 442456.710 ]
    c = read_bro(location)
    # keep 20% of cpts for testing
    c_train = c[:int(len(c)*0.8)]
    c_test = c[int(len(c)*0.8):]
    # Test first with one array
    Fr = np.array([friction_nb for cpt in c_train for friction_nb in cpt['friction_nbr'] ])
    depth = np.array([depth for cpt in c_train for depth in cpt['depth'] ])
    X = np.array([Fr, depth]).T



    # train the model
    n_components = 4
    coviarance_type = 'diag'
    n_iter = 10000
    model = hmm.GaussianHMM(n_components=n_components, covariance_type=coviarance_type, n_iter=n_iter)
    model.fit(X)
    # predict the hidden state per cpt
    for cpt in c_test:
        obs = np.array([cpt['friction_nbr'], cpt['depth']]).T
        seq = model.predict(obs)
        # plot the cpt and the hidden state
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(cpt['friction_nbr'], cpt['depth'], c=seq, cmap='viridis')
        # add labels
        plt.xlabel('friction number')
        plt.ylabel('depth')
        # add title
        plt.title('cpt: ' + cpt['name'])
        # add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=n_components))
        sm._A = []
        plt.colorbar(sm)
        # add colorbar label
        # save figure
        plt.savefig('cpt_' + cpt['name'] + '.png')




