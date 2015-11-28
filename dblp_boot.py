import redspline
from scipy import interpolate
import pickle

data = "snap_dblp.txt"
n = 317080
k = 126
boot_n = 1000

my_spec = redspline.Spectral(data, n)

for i in range(15):
    my_boot = redspline.Bootstral(my_spec, i, boot_n)
    my_boot.fit(interpolate.UnivariateSpline, k)
    my_boot.plot()
    with open('dblp_score_boot' + str(i) +'.txt', 'w') as score:
        score.write(str(my_test.score()))
    with open('dblp_dpline_boot' + str(i) +'.pickle', 'wb') as handle:
        pickle.dump(my_test, handle)