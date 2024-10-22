# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Entry points that allow users to execute GUI or Cli programs.
"""

import so4gp as sgp
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
        if allow_mp:
            msg_para = "True"
            if eval_mode:
                import ntpath
                import numpy as np
                f_name = ntpath.basename(f_path)
                f_name = f_name.replace('.csv', '')

                list_tgp, trans_data, time_data = tgp.discover_tgp(parallel=True, eval_mode=True)
                np.savetxt(f_name + '_transformed_data.csv', trans_data, fmt='%s')
                np.savetxt(f_name + '_time_data.csv', time_data, fmt='%s')
            else:
                list_tgp = tgp.discover_tgp(parallel=True)
        else:
            msg_para = "False"
            if eval_mode:
                import ntpath
                import numpy as np
                f_name = ntpath.basename(f_path)
                f_name = f_name.replace('.csv', '')

                list_tgp, trans_data, time_data = tgp.discover_tgp(parallel=True, eval_mode=True)
                np.savetxt(f_name + '_transformed_data.csv', trans_data, fmt='%s')
                np.savetxt(f_name + '_time_data.csv', time_data, fmt='%s')
            else:
                list_tgp = tgp.discover_tgp()

        if isinstance(tgp, TGradAMI):
            output_txt = "Algorithm: T-GRAANK AMI\n"
        else:
            output_txt = "Algorithm: T-GRAANK \n"
        output_txt += "No. of (dataset) attributes: " + str(tgp.col_count) + '\n'
        output_txt += "No. of (dataset) tuples: " + str(tgp.row_count) + '\n'
        output_txt += "Minimum support: " + str(min_sup) + '\n'
        output_txt += "Minimum representativity: " + str(min_rep) + '\n'
        output_txt += "Multi-core execution: " + str(msg_para) + '\n'
        output_txt += "Number of cores: " + str(tgp.cores) + '\n'
        output_txt += "Number of tasks: " + str(tgp.max_step) + '\n\n'

        for txt in tgp.titles:
            col = int(txt[0])
            if col == tgt_col:
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
    except AttributeError as error:
        output_txt = "Failed: " + str(error)
        print(error)
        return output_txt


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
