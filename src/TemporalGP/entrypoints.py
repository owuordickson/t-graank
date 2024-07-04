# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Entry points that allow users to execute GUI or Cli programs
"""


from .cli_main import terminal_app


def main_cli():
    """
    Start terminal/CMD application.
    :return:
    """
    terminal_app()
