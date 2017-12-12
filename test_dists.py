import numpy as np
from scipy.stats.kde import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

N = 1000
colors = ['red', 'blue', 'green', 'orange', 'yellow', 'violet']

data = 3.
timesteps = []
for i in xrange(6):
    data1 = np.abs(np.random.normal(loc=data, scale=3., size=N)-data)+data
    data2 = -np.abs(np.random.normal(loc=data, scale=1., size=N)-data)+data
    mask = np.random.binomial(1, 0.5, N)
    data = mask*data1 + (1-mask)*data2

    # timesteps += [sns.distplot(data, hist=False)]
    timesteps += [data]

g = sns.FacetGrid(pd.DataFrame({'timesteps': timesteps}))
g.map(sns.distplot, "timesteps", hist=False)
plt.show()
