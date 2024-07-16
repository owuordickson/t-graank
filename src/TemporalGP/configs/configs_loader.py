# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Loads default configurations from 'configs.ini' file
"""

import os
import sys
import configparser
from optparse import OptionParser
from ypstruct import struct


def load_configs():
    options_gp = struct()
    options_tgp = struct()

    # Load configuration from file
    config = configparser.ConfigParser()
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = 'configs.ini'
        config_file = os.path.join(script_dir, config_path)
        config.read(config_file)

        # 1. (configs.ini) GP Configurations
        options_gp.file_path = str(config.get('gradual-patterns', 'file_path'))
        options_gp.min_sup = float(config.get('gradual-patterns', 'min_support'))
        options_gp.num_cores = int(config.get('computation', 'cpu_cores'))
        options_gp.allow_multiprocessing = int(config.get('computation', 'allow_multiprocessing'))

        # 2. (configs.ini) TGP Configurations
        options_tgp.ref_col = int(config.get('temporal-gradual-patterns', 'ref_column'))
        options_tgp.min_rep = float(config.get('temporal-gradual-patterns', 'min_representation'))
    except configparser.NoSectionError:
        print("Default configs!")
        # 1. (Default) GP Configurations
        options_gp.file_path = ""
        options_gp.min_sup = 0.5
        options_gp.num_cores = 1
        options_gp.allow_multiprocessing = 1

        # 2. (Default) TGP Configurations
        options_tgp.ref_col = 1
        options_tgp.min_rep = 0.5

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
    optparser.add_option('-n', '--cores',
                         dest='numCores',
                         help='number of cores',
                         default=options_gp.num_cores,
                         type='int')
    optparser.add_option('-c', '--refColumn',
                         dest='refCol',
                         help='reference column',
                         default=options_tgp.ref_col,
                         type='int')
    optparser.add_option('-r', '--minRepresentativity',
                         dest='minRep',
                         help='minimum representativity',
                         default=options_tgp.min_rep,
                         type='float')
    (options, args) = optparser.parse_args()

    if (options.file is None) or options.file == '':
        print('No datasets-set filename specified, system with exit')
        print("Basic Usage: TemporalGP -f filename.csv")
        sys.exit('System will exit')

    # configs_data = {
    #    "gp_options": options_gp,
    #    "tgp_options": options_tgp,
    # }
    return options
