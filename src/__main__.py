# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU GPL v3.0
# See the LICENSE file in the root of this
# repository for complete details.

"""
A launcher for executing the application as a Terminal app.

Usage:
    $python3 __main__.py -f ../datasets/DATASET.csv -c 0 -s 0.5 -r 0.5 -p 1

Description:
    f -> file path (CSV)
    c -> reference column
    s -> minimum support
    r -> representativity
"""


from TemporalGP.cli_tgp import main_cli

if __name__ == "__main__":
    main_cli()
