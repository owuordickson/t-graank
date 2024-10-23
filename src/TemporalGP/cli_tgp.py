# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Entry points that allow users to execute GUI or Cli programs.
"""

import so4gp as sgp
from matplotlib.pyplot import xlabel

from .configs.configs_loader import load_configs
# from .TGP.t_graank import TGrad
from .TGP.tgrad_ami import TGradAMI


def execute_tgp(f_path: str, min_sup: float, tgt_col: int, min_rep: float, num_cores: int, allow_mp: bool, eq=False, eval_mode=False):
    """
    Executes T-GRAANK algorithm using the user-specified configuration options.

    :param f_path: input path of a CSV file/Pandas DataFrame.
    :param tgt_col: target/reference column of the data-set.
    :param min_sup: minimum support threshold.
    :param min_rep: minimum representativity threshold.
    :param num_cores: number of available cores.
    :param allow_mp: allow multiprocessing.
    :param eq: assign equal values as valid?
    :param eval_mode: run in 'evaluation/testing' mode?
    :return: results in string format.
    """
    try:

        if num_cores <= 1:
            num_cores = sgp.get_num_cores()

        # tgp = TGrad(f_path, eq, min_sup, tgt_col, min_rep, num_cores)
        tgp = TGradAMI(f_path, eq, min_sup, tgt_col, min_rep, num_cores)
        if eval_mode:
            list_tgp, trans_data, time_data = tgp.discover_tgp(parallel=allow_mp, eval_mode=True)
            output_txt = produce_output_txt(f_path, allow_mp, tgp, list_tgp)
            produce_eval_pdf(f_path, tgt_col, output_txt, trans_data, time_data)
        else:
            list_tgp = tgp.discover_tgp(parallel=allow_mp)
            output_txt = produce_output_txt(f_path, allow_mp, tgp, list_tgp)

        return output_txt
    except AttributeError as error:
        output_txt = "Failed: " + str(error)
        print(error)
        return output_txt


def produce_output_txt(f_path, allow_mp, tgp, list_tgp):
    """"""
    if allow_mp:
        msg_para = "True"
    else:
        msg_para = "False"

    if isinstance(tgp, TGradAMI):
        output_txt = "Algorithm: T-GRAANK AMI\n"
    else:
        output_txt = "Algorithm: T-GRAANK \n"
    output_txt += "No. of (dataset) attributes: " + str(tgp.col_count) + '\n'
    output_txt += "No. of (dataset) tuples: " + str(tgp.row_count) + '\n'
    output_txt += "Minimum support: " + str(tgp.thd_supp) + '\n'
    output_txt += "Minimum representativity: " + str(tgp.min_rep) + '\n'
    output_txt += "Multi-core execution: " + str(msg_para) + '\n'
    output_txt += "Number of cores: " + str(tgp.cores) + '\n'
    output_txt += "Number of tasks: " + str(tgp.max_step) + '\n\n'

    for txt in tgp.titles:
        col = int(txt[0])
        if col == tgp.target_col:
            output_txt += (str(txt[0]) + '. ' + str(txt[1].decode()) + '**' + '\n')
        else:
            output_txt += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

    output_txt += str("\nFile: " + f_path + '\n')
    output_txt += str("\nPattern : Support" + '\n')

    count = 0
    if isinstance(tgp, TGradAMI):
        if list_tgp:
            count = len(list_tgp)
            for tgp in list_tgp:
                output_txt += f"{tgp.to_string()} :  {tgp.support}\n"
    else:
        for obj in list_tgp:
            if obj:
                for tgp in obj:
                    count += 1
                    # output_txt += (str(tgp.to_string()) + ' : ' + str(tgp.support) +
                    #               ' | ' + str(tgp.time_lag.to_string()) + '\n')
                    output_txt += f"{tgp.to_string()} :  {tgp.support}\n"

    output_txt += "\n\n Number of patterns: " + str(count) + '\n'
    return output_txt


def produce_eval_pdf(f_path, tgt_col, out_txt, trans_data, time_data):
    """"""
    import ntpath
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    from statsmodels.tsa.seasonal import seasonal_decompose
    from matplotlib.backends.backend_pdf import PdfPages

    f_name = ntpath.basename(f_path)
    f_name = f_name.replace('.csv', '')
    pdf_file = f_name + "_results.pdf"

    data_obj = TGradAMI.process_time(trans_data)
    col_count = trans_data.shape[1]
    tgt_col =  tgt_col - (col_count-data_obj.col_count)
    num_plts = data_obj.attr_cols.shape[0] - 1
    print(f"Target column: {tgt_col}")

    # datetime_series = pd.to_datetime(data_obj.data[1:, 0].astype(float), unit='s')
    # datetime_index = pd.DatetimeIndex(datetime_series, freq='h')
    datetime_index = pd.date_range(start="2021-01-01", periods=(data_obj.row_count-1), freq="D")
    ts_1 = pd.Series(data_obj.data[1:, tgt_col], index=datetime_index)
    decomp_ts_1 = seasonal_decompose(ts_1, model='additive')
    trend_1 = np.array(decomp_ts_1.trend)
    trend_1 = trend_1[~np.isnan(trend_1)]
    max_plts = 0
    tgt_title = 'Target Col: ' + data_obj.titles[tgt_col][1].decode()
    lst_figs = []
    for col in data_obj.attr_cols:
        if col != tgt_col:
            col_title = data_obj.titles[col][1].decode()
            ts_2 = pd.Series(data_obj.data[1:, col], index=datetime_index)
            decomp_ts_2 = seasonal_decompose(ts_2, model='additive')
            trend_2 = np.array(decomp_ts_2.trend)
            trend_2 = trend_2[~np.isnan(trend_2)]
            distance, path = fastdtw(trend_1.reshape(-1, 1), trend_2.reshape(-1, 1), dist=euclidean)
            arr_path = np.array(path)
            err = np.abs(arr_path[:, 0] - arr_path[:, 1])
            avg_err = round(np.mean(err), 4)

            if max_plts > 0:
                # add plot
                max_plts -= 1
                ax.plot([p[0] for p in path], [p[1] for p in path], '-', label=f"{col_title}: {avg_err}")
                ax.legend()
            else:
                # new Figure
                max_plts = 4
                fig = plt.Figure(figsize=(5, 5))
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title('DTW Warping Path')
                ax.set(xlabel=tgt_title, ylabel='Time Series 2')
                ax.plot([p[0] for p in path], [p[1] for p in path], '-', label=f"{col_title}: {avg_err}")
                ax.legend()
                lst_figs.append(fig)

    fig_res = plt.Figure(figsize=(8.5, 11), dpi=300)
    ax_res = fig_res.add_subplot(1, 1, 1)
    ax_res.set_axis_off()
    ax_res.set_title("FTGP Results")
    ax_res.text(0, 1, out_txt, horizontalalignment='left', verticalalignment='top', transform=ax_res.transAxes)
    with (PdfPages(pdf_file)) as pdf:
        pdf.savefig(fig_res)
        for fig in lst_figs:
            pdf.savefig(fig)

    np.savetxt(f_name + '_transformed_data.csv', trans_data[:, data_obj.attr_cols], fmt='%s', delimiter=',')
    np.savetxt(f_name + '_timestamp_data.csv', time_data, fmt='%s', delimiter=',')
    return lst_figs


def main_cli():
    """
        Initializes and starts terminal/CMD application.
        :return:
    """
    import time
    # import tracemalloc

    cfg = load_configs()

    start = time.time()
    # tracemalloc.start()
    res_text = execute_tgp(cfg.file, cfg.minSup, cfg.tgtCol, cfg.minRep, cfg.numCores, cfg.allowPara, eval_mode=cfg.evalMode)
    # snapshot = tracemalloc.take_snapshot()
    end = time.time()

    wr_text = ("Run-time: " + str(end - start) + " seconds\n")
    # wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
    wr_text += str(res_text)
    f_name = str('res_tgp' + str(end).replace('.', '', 1) + '.txt')
    sgp.write_file(wr_text, f_name, True)
    print(wr_text)
