import redspline
from scipy import interpolate
import pickle

data = "snap_dblp.txt"
n = 317080
k = 126

my_test = redspline.Spectral(data, n)

my_test.fit(interpolate.UnivariateSpline, k)
my_test.plot()

with open('dblp_score.txt', 'w') as score:
    score.write(str(my_test.score())

with open('dblp_dpline.pickle', 'wb') as handle:
  pickle.dump(my_test, handle)