from TemporalGP.TGP.tgrad_ami import TGradAMI
from TemporalGP.TGP.t_graank import TGrad


# f_path = "../datasets/DATASET.csv"
# f_path = "../datasets/rain_temp2013-2015.csv"
# f_path = "../datasets/air_quality25.csv"
# f_path = "../datasets/air_quality1k.csv"
f_path = "../datasets/ke_rain_data_2k.csv"

eq = False
min_sup = 0.5
tgt_col = 1
min_rep = 0.75
mi_err_margin = 0.0001
eval_mode = True
clustering_method = False

#t_grad = TGradAMI(f_path, min_sup, eq, target_col=tgt_col, min_rep=min_rep, min_error=mi_err_margin)
t_grad = TGrad(f_path, min_sup, eq, target_col=tgt_col, min_rep=min_rep)
eval_dict = t_grad.discover_tgp()
print(eval_dict)
