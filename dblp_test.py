import redspline
from scipy import interpolate
import pickle

data = "snap_dblp.txt"
n = 317080
k = 126

my_test = redspline.Spectral(data, n)

my_test.fit(interpolate.UnivariateSpline, k)
my_test.plot()
my_test.score()

with open('dblp_dpline.pickle', 'wb') as handle:
  pickle.dump(my_test, handle)