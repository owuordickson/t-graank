# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0
# See the LICENSE file in the root of this
# repository for complete details.

"""
Entry points that allow users to execute GUI or Cli programs.
"""

import sys
import json
from optparse import OptionParser
import so4gp as sgp

from .configs.configs_loader import load_configs
from .TGP.t_graank import TGrad
from .TGP.tgrad_ami import TGradAMI


def execute_tgp(f_path: str, min_sup: float, tgt_col: int, min_rep: float, min_error: float, num_cores: int,
                allow_mp: bool, eq: bool=False, allow_clustering: bool=False, eval_mode=False, algorithm: int=0):
    """
    Executes T-GRAANK algorithm using the user-specified configuration options

    :param f_path: input path of a CSV file/Pandas DataFrame
    :param tgt_col: target/reference column of the data-set
    :param min_sup: minimum support threshold
    :param min_rep: minimum representativity threshold
    :param min_error: minimum mutual information error threshold
    :param num_cores: number of available cores
    :param allow_mp: allow multiprocessing
    :param eq: to assign equal values as valid
    :param allow_clustering: using clustering method to estimate time delays
    :param eval_mode: run in 'evaluation/testing' mode
    :param algorithm: algorithm to use (T-Grad or TGradAMI)
    :return: results in string format
    """
    try:
        if num_cores <= 1:
            num_cores = sgp.get_num_cores()

        if algorithm == 0:
            t_grad = TGrad(f_path, min_sup, eq, target_col=tgt_col, min_rep=min_rep)
            res = t_grad.discover_tgp(parallel=allow_mp, num_cores=num_cores)
            res_dict = json.loads(res)
        elif algorithm == 1:
            t_grad = TGradAMI(f_path, min_sup, eq, target_col=tgt_col, min_rep=min_rep, min_error=min_error)
            res_dict = t_grad.discover_tgp(use_clustering=allow_clustering, eval_mode=eval_mode)
        else:
            return "Invalid algorithm specified"

        print(res_dict)
        # produce_eval_pdf(f_path, tgt_col, output_txt, trans_data, time_data)
    except ZeroDivisionError as error:
        print("Failed: " + str(error))


def main_cli():
    """
        Initializes and starts a terminal / CMD application.
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
    optparser.add_option('-m', '--minMIError',
                         dest='minError',
                         help='minimum mutual information error',
                         default=options_tgp.min_mi_error,
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

    # import tracemalloc
    # tracemalloc.start()
    execute_tgp(cfg.file, cfg.minSup, cfg.tgtCol, cfg.minRep, cfg.minError, cfg.numCores, cfg.allowPara,
                           allow_clustering=cfg.useClusters, eval_mode=cfg.evalMode)
    # snapshot = tracemalloc.take_snapshot()
