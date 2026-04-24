# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0
# See the LICENSE file in the root of this
# repository for complete details.

"""
Loads default configurations from the 'configs.ini' file
"""

import os
import configparser


def load_configs():
    options_gp = {"file_path": "", "min_sup": 0.5, "num_cores": 1, "allow_multiprocessing": 1, "eval_mode": 0,}
    options_tgp = {"target_column": 1, "min_rep": 0.5, "min_mi_error": 0.1, "use_clustering": 0,}

    # Load configuration from file
    config = configparser.ConfigParser()
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = 'configs.ini'
        config_file = os.path.join(script_dir, config_path)
        config.read(config_file)

        # 1. (configs.ini) GP Configurations
        options_gp["file_path"] = str(config.get('gradual-patterns', 'file_path'))
        options_gp["min_sup"] = float(config.get('gradual-patterns', 'min_support'))
        options_gp["num_cores"] = int(config.get('computation', 'cpu_cores'))
        options_gp["allow_multiprocessing"] = int(config.get('computation', 'allow_multiprocessing'))
        options_gp["eval_mode"] = int(config.get('computation', 'eval_mode'))

        # 2. (configs.ini) TGP Configurations
        options_tgp["target_column"] = int(config.get('temporal-gradual-patterns', 'target_column'))
        options_tgp["min_rep"] = float(config.get('temporal-gradual-patterns', 'min_representation'))
        options_tgp["min_mi_error"] = float(config.get('temporal-gradual-patterns', 'mi_error_margin'))
        options_tgp["use_clustering"] = int(config.get('temporal-gradual-patterns', 'use_clustering_method'))
        return options_gp, options_tgp
    except configparser.NoSectionError:
        print("Default configs!")
        return options_gp, options_tgp
