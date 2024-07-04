import os
import pandas as pd
import statistics
from tabulate import tabulate


# -------- OTHER METHODS -----------


def analyze_gps(data_src, min_sup, est_gps, approach='bfs'):
    """Description

    For each estimated GP, computes its true support using GRAANK approach and returns the statistics (% error,
    and standard deviation).

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> columns = ['Age', 'Salary', 'Cars', 'Expenses']
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> estimated_gps = list()
    >>> temp_gp = sgp.ExtGP()
    >>> temp_gp.add_items_from_list(['0+', '1-'])
    >>> temp_gp.set_support(0.5)
    >>> estimated_gps.append(temp_gp)
    >>> temp_gp = sgp.ExtGP()
    >>> temp_gp.add_items_from_list(['1+', '3-', '0+'])
    >>> temp_gp.set_support(0.48)
    >>> estimated_gps.append(temp_gp)
    >>> res = sgp.analyze_gps(dummy_df, min_sup=0.4, est_gps=estimated_gps, approach='bfs')
    >>> print(res)
    Gradual Pattern       Estimated Support    True Support  Percentage Error      Standard Deviation
    ------------------  -------------------  --------------  ------------------  --------------------
    ['0+', '1-']                       0.5              0.4  25.0%                              0.071
    ['1+', '3-', '0+']                 0.48             0.6  -20.0%                             0.085

    :param data_src: data set file

    :param min_sup: minimum support (set by user)
    :type min_sup: float

    :param est_gps: estimated GPs
    :type est_gps: list

    :param approach: 'bfs' (default) or 'dfs'
    :type approach: str

    :return: tabulated results
    """
    if approach == 'dfs':
        d_set = DataGP(data_src, min_sup)
        d_set.fit_tids()
    else:
        d_set = DataGP(data_src, min_sup)
        d_set.fit_bitmap()
    headers = ["Gradual Pattern", "Estimated Support", "True Support", "Percentage Error", "Standard Deviation"]
    data = []
    for est_gp in est_gps:
        est_sup = est_gp.support
        est_gp.set_support(0)
        if approach == 'dfs':
            true_gp = est_gp.validate_tree(d_set)
        else:
            true_gp = est_gp.validate_graank(d_set)
        true_sup = true_gp.support

        if true_sup == 0:
            percentage_error = np.inf
            st_dev = np.inf
        else:
            percentage_error = ((est_sup - true_sup) / true_sup) * 100
            st_dev = statistics.stdev([est_sup, true_sup])

        if len(true_gp.gradual_items) == len(est_gp.gradual_items):
            data.append([est_gp.to_string(), round(est_sup, 3), round(true_sup, 3), str(round(percentage_error, 3))+'%',
                         round(st_dev, 3)])
        else:
            data.append([est_gp.to_string(), round(est_sup, 3), -1, np.inf, np.inf])
    return tabulate(data, headers=headers)


def get_num_cores():
    """Description

    Finds the count of CPU cores in a computer or a SLURM super-computer.
    :return: number of cpu cores (int)
    """
    num_cores = get_slurm_cores()
    if not num_cores:
        num_cores = mp.cpu_count()
    return num_cores


def get_slurm_cores():
    """Description

    Test computer to see if it is a SLURM environment, then gets number of CPU cores.
    :return: count of CPUs (int) or False
    """
    try:
        cores = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        return cores
    except ValueError:
        try:
            str_cores = str(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            temp = str_cores.split('(', 1)
            cpus = int(temp[0])
            str_nodes = temp[1]
            temp = str_nodes.split('x', 1)
            str_temp = str(temp[1]).split(')', 1)
            nodes = int(str_temp[0])
            cores = cpus * nodes
            return cores
        except ValueError:
            return False
    except KeyError:
        return False


def write_file(data, path, wr=True):
    """Description

    Writes data into a file
    :param data: information to be written
    :param path: name of file and storage path
    :param wr: writes data into file if True
    :return:
    """
    if wr:
        with open(path, 'w') as f:
            f.write(data)
            f.close()
    else:
        pass
