from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import tensorboard as tb
from scipy.signal import savgol_filter
import numpy as np

def calculateLowerUpper(column):
    mean = np.mean(column.to_numpy())
    std = np.std(column.to_numpy())
    lower = (mean - 1.96 * std) 
    upper = (mean + 1.96 * std) 
    return lower,upper


experiment_id = "a0FwRRwJQsqHOxKeqFpxnA"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()

dfw = experiment.get_scalars(pivot=True)
dfw = dfw[:12000]
print(1)

yhat_disc_loss = savgol_filter(dfw["disc_loss"], 200, 3)
lower_disc_loss, upper_disc_loss = calculateLowerUpper(dfw["disc_loss"])
yhat_gen_total_loss = savgol_filter(dfw["gen_total_loss"], 600, 3) 
lower_gen_total_loss, upper_gen_total_loss = calculateLowerUpper(dfw["gen_total_loss"])
yhat_gen_gan_loss = savgol_filter(dfw["gen_gan_loss"], 600, 3) 
lower_gen_gan_loss, upper_gen_gan_loss = calculateLowerUpper(dfw["gen_gan_loss"])

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times'], 'size': 14})
rc('text', usetex=True)

fig, axs = plt.subplots(ncols=3) #, figsize=(10, 3.2))
fig.set_size_inches(10, 3.5)

axs[0].plot(dfw["step"], yhat_disc_loss)
#axs[0].fill_between(dfw["step"], lower_disc_loss, upper_disc_loss,
#                 color='gray', alpha=0.2)
axs[1].plot(dfw["step"], yhat_gen_total_loss)
#axs[1].fill_between(dfw["step"], lower_gen_total_loss, upper_gen_total_loss,
#                 color='gray', alpha=0.2)
axs[2].plot(dfw["step"], yhat_gen_gan_loss)
axs[1].set_ylabel("Loss generator")
axs[0].set_ylabel("Loss discriminator")
axs[2].set_ylabel("Total loss GAN framework")
axs[0].set_xlabel("Steps during training")
axs[1].set_xlabel("Steps during training")
axs[2].set_xlabel("Steps during training")
axs[0].grid()
axs[1].grid()
axs[2].grid()
plt.show()



