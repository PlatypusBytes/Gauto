import numpy as np
import pandas as pd
import random
import scipy.misc

def binary_sampler_by_columns(values,dimesions, miss_rate):
  """Sample binary random variables by columns.

  Args:
    - values: all values
    - dimesions: dimension of depth and width
    - miss_rate: percentatage of missing data

  Returns:
    - binary_random_matrix: generated binary random matrix.
  """
  dim_choice = np.sqrt(dimesions)
  data_m = []
  for slice in values:  
    random_indexes = random.sample(range(0,int(dim_choice)), int(miss_rate*dim_choice))
    transposed_slice = np.reshape(list(slice), (256,256))
    ones_slice = np.ones(transposed_slice.shape)
    ones_slice[random_indexes]=0
    data_m.append(ones_slice.flatten())
  return np.array(data_m)

def binary_sampler_by_rows(values,dimesions, miss_rate):
  """Sample binary random variables by rows.

  Args:
    - values: all values
    - dimesions: dimension of depth and width
    - miss_rate: percentatage of missing data

  Returns:
    - binary_random_matrix: generated binary random matrix.
  """
  dim_choice = np.sqrt(dimesions)
  data_m = []
  for slice in values:  
    random_indexes = random.sample(range(0,int(dim_choice)), int(miss_rate*dim_choice))
    transposed_slice = np.reshape(list(slice), (256,256))
    ones_slice = np.ones(transposed_slice.shape)
    ones_slice[:, random_indexes] = 0
    data_m.append(ones_slice.flatten())
  return np.array(data_m)


def data_loader(data_name, miss_rate, value_name, sample_vertically=True):
    """Loads datasets and introduce missingness.

    Args:
      - data_name: letter, spam, or mnist
      - miss_rate: the probability of missing components

    Returns:
      data_x: original data
      miss_data_x: data with missing values
      data_m: indicator matrix for missing components
    """

    # Load data
    data_x = []
    file_name = "data/" + data_name + ".csv"
    df = pd.read_csv(file_name)
    grouped = df.groupby('z')
    for name, group in grouped:
      data_x.append( list(group[value_name]))
    data_x = np.array(data_x, dtype=float)
    # Parameters
    no, dim = data_x.shape
    if sample_vertically:
      data_m=binary_sampler_by_columns(data_x, dim, miss_rate)
    else:
      data_m=binary_sampler_by_rows(data_x, dim, miss_rate)      
    # Introduce missing data
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0.] = 0

    return data_x, miss_data_x, data_m



# MAX_QT = 10.5
# MAX_FS = 111.5
# full_data, train_input, binary_missing_data = data_loader("synthetic_data_G2", 0, "fs")
# #for counter, image_array in enumerate(train_input):
# #  scipy.misc.toimage(image_array, cmin=0.0, cmax=MAX_QT).save(f'images/{counter}.jpg')
# from PIL import Image
# # convert image to a numpy array
# for counter, image_array in enumerate(train_input):
#   image_array = np.reshape(image_array, (256, 256))
#   im = Image.fromarray((image_array/ MAX_FS) * 255)
#   im = im.convert("L")
#   im.save(f'gauto\gans\pix2pix\images/fs_{counter}.jpeg')

  
