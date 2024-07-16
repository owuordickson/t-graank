# -*- coding: utf-8 -*-
"""
Usage:
    $python3 cli_main.py -f ../datasets/DATASET.csv -c 0 -s 0.5 -r 0.5 -p 1

Description:
    f -> file path (CSV)
    c -> reference column
    s -> minimum support
    r -> representativity
"""


import so4gp as sgp
from .configs.configs_loader import load_configs
from .TGP.t_graank import TGrad


def execute_tgp(f_path: str, min_sup: float, ref_col: int, min_rep: float, num_cores: int, allow_mp: bool, eq=False):
    """
    Executes T-GRAANK algorithm using the user-specified configuration options.

    :param f_path: input path of a CSV file/Pandas DataFrame.
    :param ref_col: reference column of the data-set.
    :param min_sup: minimum support threshold.
    :param min_rep: minimum representativity threshold.
    :param num_cores: number of available cores.
    :param allow_mp: allow multiprocessing.
    :param eq: assign equal values as valid?
    :return: results in string format.
    """
    try:

        if num_cores <= 1:
            num_cores = sgp.get_num_cores()

        tgp = TGrad(f_path, eq, min_sup, ref_col, min_rep, num_cores)
        if allow_mp:
            msg_para = "True"
            list_tgp = tgp.discover_tgp(parallel=True)
        else:
            msg_para = "False"
            list_tgp = tgp.discover_tgp()

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
            if col == ref_col:
                output_txt += (str(txt[0]) + '. ' + str(txt[1].decode()) + '**' + '\n')
            else:
                output_txt += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        output_txt += str("\nFile: " + f_path + '\n')
        output_txt += str("\nPattern : Support" + '\n')

        count = 0
        for obj in list_tgp:
            if obj:
                for tgp in obj:
                    count += 1
                    output_txt += (str(tgp.to_string()) + ' : ' + str(tgp.support) +
                                   ' | ' + str(tgp.time_lag.to_string()) + '\n')

        output_txt += "\n\n Number of patterns: " + str(count) + '\n'
        return output_txt
    except Exception as error:
        output_txt = "Failed: " + str(error)
        print(error)
        return output_txt


def terminal_app():
    """
        Initializes and executes T-GRAANK algorithm.
        :return:
    """
    import time
    # import tracemalloc

    cfg = load_configs()

    start = time.time()
    # tracemalloc.start()
    res_text = execute_tgp(cfg.file, cfg.minSup, cfg.refCol, cfg.minRep, cfg.numCores, cfg.allowPara)
    # snapshot = tracemalloc.take_snapshot()
    end = time.time()

    wr_text = ("Run-time: " + str(end - start) + " seconds\n")
    # wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
    wr_text += str(res_text)
    f_name = str('res_tgp' + str(end).replace('.', '', 1) + '.txt')
    sgp.write_file(wr_text, f_name, True)
    print(wr_text)
