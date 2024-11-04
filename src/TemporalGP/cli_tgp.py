# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Entry points that allow users to execute GUI or Cli programs.
"""

import sys
from optparse import OptionParser
import so4gp as sgp

from .configs.configs_loader import load_configs
from .TGP.t_graank import TGrad
from .TGP.tgrad_ami import TGradAMI


def execute_tgp(f_path: str, min_sup: float, tgt_col: int, min_rep: float, num_cores: int, allow_mp: bool, eq=False,
                allow_clustering=False, eval_mode=False):
    """
    Executes T-GRAANK algorithm using the user-specified configuration options.

    :param f_path: input path of a CSV file/Pandas DataFrame.
    :param tgt_col: target/reference column of the data-set.
    :param min_sup: minimum support threshold.
    :param min_rep: minimum representativity threshold.
    :param num_cores: number of available cores.
    :param allow_mp: allow multiprocessing.
    :param eq: assign equal values as valid?
    :param allow_clustering: using clustering method to estimate time delays?
    :param eval_mode: run in 'evaluation/testing' mode?
    :return: results in string format.
    """
    try:

        if num_cores <= 1:
            num_cores = sgp.get_num_cores()

        #tgp = TGrad(f_path, eq, min_sup, tgt_col, min_rep, num_cores)
        tgp = TGradAMI(f_path, eq, min_sup, tgt_col, min_rep, num_cores)
        if eval_mode and isinstance(tgp, TGradAMI):
            list_tgp, trans_data, time_data, gp_components = tgp.discover_tgp(parallel=allow_mp,
                                                                              use_clustering=allow_clustering,
                                                                              eval_mode=True)
            output_txt = produce_output_txt(f_path, allow_mp, allow_clustering, tgp, list_tgp)
            # produce_eval_pdf(f_path, tgt_col, output_txt, trans_data, time_data)
        else:
            list_tgp = tgp.discover_tgp(parallel=allow_mp, use_clustering=allow_clustering)
            output_txt = produce_output_txt(f_path, allow_mp, allow_clustering, tgp, list_tgp)

        return output_txt
    except ZeroDivisionError as error:
        output_txt = "Failed: " + str(error)
        print(error)
        return output_txt


def produce_output_txt(f_path, allow_mp, allow_clustering, tgp, list_tgp):
    """"""
    if allow_mp:
        msg_para = "True"
    else:
        msg_para = "False"

    if isinstance(tgp, TGradAMI):
        output_txt = f"Algorithm: T-GRAANK AMI {'(with KMeans & Hill-climbing)' if allow_clustering else '(with Slide-Recalculate)'}\n"
    else:
        output_txt = "Algorithm: T-GRAANK (with Slide-Recalculate)\n"
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
                gp_str = f"{tgp.to_string()} :  {tgp.support}"
                if len(gp_str) > 100:
                    gp_str = gp_str[:100] + '\n' + gp_str[100:]
                output_txt += f"{gp_str}\n"
    else:
        for obj in list_tgp:
            if obj:
                for tgp in obj:
                    count += 1
                    # output_txt += (str(tgp.to_string()) + ' : ' + str(tgp.support) +
                    #               ' | ' + str(tgp.time_lag.to_string()) + '\n')
                    gp_str = f"{tgp.to_string()} :  {tgp.support}"
                    if len(gp_str) > 100:
                        gp_str = gp_str[:100] + '\n' + gp_str[100:]
                    output_txt += f"{gp_str}\n"

    output_txt += "\n\n Number of patterns: " + str(count) + '\n'
    return output_txt


def produce_eval_pdf(f_path, tgt_col, out_txt, trans_data, time_data):
    """"""
    import time
    import ntpath
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from numpy.linalg import norm
    # from fastdtw import fastdtw
    # from scipy.spatial.distance import euclidean
    from statsmodels.tsa.seasonal import seasonal_decompose
    from matplotlib.backends.backend_pdf import PdfPages

    file_stamp = time.time()
    f_name = ntpath.basename(f_path)
    f_name = f_name.replace('.csv', '')
    pdf_file = f_name + str(file_stamp).replace('.', '', 1) + "_results.pdf"

    data_obj = TGradAMI.process_time(trans_data)
    col_count = trans_data.shape[1]
    tgt_col =  tgt_col - (col_count-data_obj.col_count)
    # num_plts = data_obj.attr_cols.shape[0] - 1

    # datetime_series = pd.to_datetime(data_obj.data[1:, 0].astype(float), unit='s')
    # datetime_index = pd.DatetimeIndex(datetime_series, freq='h')
    datetime_index = pd.date_range(start="2021-01-01", periods=(data_obj.row_count-1), freq="D")
    ts_1 = pd.Series(data_obj.data[1:, tgt_col], index=datetime_index)
    decomp_ts_1 = seasonal_decompose(ts_1, model='additive')
    # trend_1 = np.array(decomp_ts_1.trend)
    seasonal_1 = np.array(decomp_ts_1.seasonal)
    # trend_1 = trend_1[~np.isnan(trend_1)]
    seasonal_1 = seasonal_1[~np.isnan(seasonal_1)]
    # max_plts = 0
    # tgt_title = 'Target Col: ' + data_obj.titles[tgt_col][1].decode()
    tgt_title = data_obj.titles[tgt_col][1].decode() + '*'
    lst_res = []
    sim_txt = ""
    for col in data_obj.attr_cols:
        if col != tgt_col:
            col_title = data_obj.titles[col][1].decode()
            ts_2 = pd.Series(data_obj.data[1:, col], index=datetime_index)
            decomp_ts_2 = seasonal_decompose(ts_2, model='additive')
            # trend_2 = np.array(decomp_ts_2.trend)
            seasonal_2 = np.array(decomp_ts_2.seasonal)
            # trend_2 = trend_2[~np.isnan(trend_2)]
            seasonal_2 = seasonal_2[~np.isnan(seasonal_2)]
            cos_similarity = np.dot(seasonal_1.ravel(), seasonal_2.ravel()) / (norm(seasonal_1.ravel()) * norm(seasonal_2.ravel()))
            sim_txt += f"{tgt_title} - {col_title}: {round(cos_similarity, 4)}\n"
            lst_res.append(f"{tgt_title} - {col_title}: {round(cos_similarity, 4)}")

            """
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
                fig = plt.Figure(figsize=(6, 5))
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title('DTW Warping Path')
                ax.set(xlabel=tgt_title, ylabel='Time Series 2')
                ax.plot([p[0] for p in path], [p[1] for p in path], '-', label=f"{col_title}: {avg_err}")
                ax.legend()
                lst_figs.append(fig)
    """

    fig_res = plt.Figure(figsize=(8.5, 11), dpi=300)
    ax_res = fig_res.add_subplot(1, 1, 1)
    ax_res.set_axis_off()
    ax_res.set_title("FTGP Results")
    ax_res.text(0, 1, out_txt, horizontalalignment='left', verticalalignment='top', transform=ax_res.transAxes)

    fig_res1 = plt.Figure(figsize=(8.5, 11), dpi=300)
    ax_res1 = fig_res1.add_subplot(1, 1, 1)
    ax_res1.set_axis_off()
    ax_res1.set_title("Cosine Similarity of Seasonal Trends")
    ax_res1.text(0, 1, sim_txt, horizontalalignment='left', verticalalignment='top', transform=ax_res1.transAxes)

    with (PdfPages(pdf_file)) as pdf:
        pdf.savefig(fig_res)
        pdf.savefig(fig_res1)
        # for fig in lst_figs:
        #    pdf.savefig(fig)

    np.savetxt(f_name + str(file_stamp).replace('.', '', 1) +'_transformed_data.csv', trans_data[:, data_obj.attr_cols], fmt='%s', delimiter=',')
    np.savetxt(f_name + str(file_stamp).replace('.', '', 1) +'_timestamp_data.csv', time_data, fmt='%s', delimiter=',')
    return lst_res


def main_cli():
    """
        Initializes and starts terminal/CMD application.
        :return:
    """
    options_gp, options_tgp = load_configs()

    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='file',
                         help='path to file containing csv',
                         default=options_gp.file_path,
                         type='string')
    optparser.add_option('-s', '--minSupport',
                         dest='minSup',
                         help='minimum support value',
                         default=options_gp.min_sup,
                         type='float')
    optparser.add_option('-p', '--allowMultiprocessing',
                         dest='allowPara',
                         help='allow multiprocessing',
                         default=options_gp.allow_multiprocessing,
                         type='int')
    optparser.add_option('-x', '--evaluationMode',
                         dest='evalMode',
                         help='run in evaluation mode',
                         default=options_gp.eval_mode,
                         type='int')
    optparser.add_option('-c', '--cores',
                         dest='numCores',
                         help='number of cores',
                         default=options_gp.num_cores,
                         type='int')
    optparser.add_option('-t', '--targetColumn',
                         dest='tgtCol',
                         help='target column',
                         default=options_tgp.target_column,
                         type='int')
    optparser.add_option('-r', '--minRepresentativity',
                         dest='minRep',
                         help='minimum representativity',
                         default=options_tgp.min_rep,
                         type='float')
    optparser.add_option('-k', '--useClustering',
                         dest='useClusters',
                         help='use clustering method',
                         default=options_tgp.use_clustering,
                         type='int')
    (cfg, args) = optparser.parse_args()
    cfg.useClusters = bool(cfg.useClusters)
    cfg.evalMode = bool(cfg.evalMode)

    if (cfg.file is None) or cfg.file == '':
        print('No datasets-set filename specified, system with exit')
        print("Basic Usage: TemporalGP -f filename.csv")
        sys.exit('System will exit')

    import time
    # import tracemalloc

    start = time.time()
    # tracemalloc.start()
    res_text = execute_tgp(cfg.file, cfg.minSup, cfg.tgtCol, cfg.minRep, cfg.numCores, cfg.allowPara,
                           allow_clustering=cfg.useClusters, eval_mode=cfg.evalMode)
    # snapshot = tracemalloc.take_snapshot()
    end = time.time()

    wr_text = ("Run-time: " + str(end - start) + " seconds\n")
    # wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
    wr_text += str(res_text)
    f_name = str('res_tgp' + str(end).replace('.', '', 1) + '.txt')
    sgp.write_file(wr_text, f_name, True)
    print(wr_text)
