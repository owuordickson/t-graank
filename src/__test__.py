import numpy as np
from TemporalGP.TGP.tgrad_ami import TGradAMI


# f_path = "../../datasets/rain_temp2013-2015.csv"
f_path = "../../datasets/air_quality25.csv"

tgp = TGradAMI(f_path, False, 0.5, 2, 1, 1)
tgp.discover_tgp()
print(f"Initial MIs: {tgp.initial_mutual_info} \n")
print(f"Delayed MIs: {tgp.mi_arr[:, 0]} \n")

# vec1 = np.array(tgp.initial_mutual_info).reshape(1, -1)
# diff_vec = tgp.mi_arr[:, 0] - tgp.initial_mutual_info[0]
# var = np.var(diff_vec, axis=0)
init_mi = np.full(tgp.mi_arr[:, 0].shape, tgp.initial_mutual_info[0])
squared_diff = (tgp.mi_arr[:, 0] - tgp.initial_mutual_info[0]) ** 2
absolute_error = np.sqrt(squared_diff)

print(f"Init MI: {init_mi}\n")
print(f"Diff: {squared_diff}\n")
print(f"Abs.E.: {absolute_error}\n")
print(f"Optimal Step: {np.argmin(absolute_error)}\n")
