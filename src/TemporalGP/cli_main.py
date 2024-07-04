# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent and Joseph Orero"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "19 November 2019"

Usage:
    $python3 cli_main.py -f ../datasets/DATASET.csv -c 0 -s 0.5 -r 0.5 -p 1

Description:
    f -> file path (CSV)
    c -> reference column
    s -> minimum support
    r -> representativity

"""

import sys
from optparse import OptionParser
from .TGP.t_graank import TGrad
# from .TGP.t_graank_h5 import TGrad_5


def tgp_app(f_path, ref_item, min_sup, min_rep, num_cores, eq=False):
    try:

        tgp = TGrad(f_path, eq, min_sup, ref_item, min_rep, num_cores)
        if num_cores >= 1:
            msg_para = "True"
            list_tgp = tgp.discover_tgp(parallel=True)
        else:
            msg_para = "False"
            list_tgp = tgp.discover_tgp()

        wr_line = "Algorithm: T-GRAANK \n"
        wr_line += "No. of (dataset) attributes: " + str(tgp.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(tgp.row_count) + '\n'
        wr_line += "Minimum support: " + str(min_sup) + '\n'
        wr_line += "Minimum representativity: " + str(min_rep) + '\n'
        wr_line += "Multi-core execution: " + str(msg_para) + '\n'
        wr_line += "Number of cores: " + str(tgp.cores) + '\n'
        wr_line += "Number of tasks: " + str(tgp.max_step) + '\n\n'

        for txt in tgp.titles:
            col = int(txt[0])
            if col == ref_item:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '**' + '\n')
            else:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        count = 0
        for obj in list_tgp:
            if obj:
                for tgp in obj:
                    count += 1
                    wr_line += (str(tgp.to_string()) + ' : ' + str(tgp.support) +
                                ' | ' + str(tgp.time_lag.to_string()) + '\n')

        wr_line += "\n\n Number of patterns: " + str(count) + '\n'
        return wr_line
    except Exception as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line


def tgp_app_h5(f_path, ref_item, min_sup, min_rep, allow_para, eq=False):
    try:
        tgp = TGrad_5(f_path, eq, min_sup, ref_item, min_rep, allow_para)
        if allow_para >= 1:
            msg_para = "True"
            list_tgp = tgp.discover_tgp(parallel=True)
        else:
            msg_para = "False"
            list_tgp = tgp.discover_tgp()

        d_set = tgp.d_set
        wr_line = "Algorithm: T-GRAANK \n"
        wr_line += "   - H5Py implementation \n"
        wr_line += "No. of (dataset) attributes: " + str(d_set.column_size) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(d_set.size) + '\n'
        wr_line += "Minimum support: " + str(min_sup) + '\n'
        wr_line += "Minimum representativity: " + str(min_rep) + '\n'
        wr_line += "Multi-core execution: " + str(msg_para) + '\n'
        wr_line += "Number of cores: " + str(tgp.cores) + '\n'
        wr_line += "Number of tasks: " + str(tgp.max_step) + '\n\n'

        for txt in d_set.title:
            col = int(txt[0])
            if col == ref_item:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '**' + '\n')
            else:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        count = 0
        for obj in list_tgp:
            if obj:
                for tgp in obj:
                    count += 1
                    wr_line += (str(tgp.to_string()) + ' : ' + str(tgp.support) +
                                ' | ' + str(tgp.time_lag.to_string()) + '\n')

        wr_line += "\n\n Number of patterns: " + str(count) + '\n'
        return wr_line
    except Exception as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line


def write_file(data, path):
    with open(path, 'w') as f:
        f.write(data)
        f.close()


def terminal_app():
    if not sys.argv:
        # pType = sys.argv[1]
        file_path = sys.argv[1]
        ref_col = sys.argv[2]
        min_sup = sys.argv[3]
        min_rep = sys.argv[4]
        allow_p = sys.argv[5]
    else:
        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             # default=None,
                             # default='../datasets/DATASET2.csv',
                             default='../datasets/rain_temp2013-2015.csv',
                             # default='../datasets/Directio.csv',
                             type='string')
        optparser.add_option('-c', '--refColumn',
                             dest='refCol',
                             help='reference column',
                             default=1,
                             type='int')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=0.5,
                             type='float')
        optparser.add_option('-r', '--minRepresentativity',
                             dest='minRep',
                             help='minimum representativity',
                             default=0.5,
                             type='float')
        optparser.add_option('-p', '--allowMultiprocessing',
                             dest='allowPara',
                             help='allow multiprocessing',
                             default=1,
                             type='int')
        (options, args) = optparser.parse_args()

        if options.file is None:
            print('No datasets-set filename specified, system with exit')
            print("Usage: $python3 cli_main.py -f filename.csv -c refColumn -s minSup  -r minRep")
            sys.exit('System will exit')

        file_path = options.file
        ref_col = options.refCol
        min_sup = options.minSup
        min_rep = options.minRep
        allow_p = options.allowPara

    import time
    # import tracemalloc

    start = time.time()
    # tracemalloc.start()
    res_text = tgp_app(file_path, ref_col, min_sup, min_rep, allow_p)
    # snapshot = tracemalloc.take_snapshot()
    end = time.time()

    wr_text = ("Run-time: " + str(end - start) + " seconds\n")
    # wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
    wr_text += str(res_text)
    f_name = str('res_temp' + str(end).replace('.', '', 1) + '.txt')
    write_file(wr_text, f_name)
    print(wr_text)
