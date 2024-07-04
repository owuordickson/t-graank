

# -------- DATA SET PREPARATION -------------
class DataGP:
    """Description of class DataGP

    A class for creating data-gp objects. A data-gp object is meant to store all the parameters required by GP
    algorithms to extract gradual patterns (GP). It takes a numeric file (in CSV format) as input and converts it into
    an object whose attributes are used by algorithms to extract GPs.

        A GP is a set of gradual items (GI) and its quality is measured by its computed support value. For example given
    a data set with 3 columns (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-} with a support
    of 0.8. This implies that 8 out of 10 objects have the values of column age 'increasing' and column 'salary'
    decreasing.

    The class provides the following attributes:
        thd_supp: minimum support threshold

        equal: eq value

        titles: column names of data source

        data: all the objects organized into their respective column

        row_count: number of objects

        col_count: number of all columns

        time_cols: column indices of the columns with data-time objects

        attr_cols: column indices of the columns with numeric values

        valid_bins: valid bitmaps (in the form of ndarray) of all gradual items corresponding to the attr_cols,
        a bitmap is valid if its computed support is equal or greater than the minimum support threshold

        no_bins: True if all none of the attr_cols yields a valid bitmap

        gradual_patterns: list of GP objects


    The class provides the following functions:
        get_attr_cols: retrieves all the columns with data that is numeric and not date-tme

        get_time_cols: retrieves the columns with date-time values

        get_gi_bitmap: computes and returns the bitmap matrix corresponding to a GI

        fit_bitmap: generates all the bitmap matrices of valid GIs

        fit_tids: generates all the transaction ids of valid GIs

        read (static): reads contents of a CSV file or data-frame

        test_time (static): tests if a str represents a date-time variable

        clean_data (static): cleans data (missing values, outliers) before extraction of GPs.

    >>> import so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> columns = ['Age', 'Salary', 'Cars', 'Expenses']
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> data_gp = sgp.DataGP(data_source=dummy_df, min_sup=0.5)
    >>> data_gp.fit_bitmap()


    """

    def __init__(self, data_source, min_sup=MIN_SUPPORT, eq=False):
        """Description of class DataGP


        A class for creating data-gp objects. A data-gp object is meant to store all the parameters required by GP
        algorithms to extract gradual patterns (GP). It takes a numeric file (in CSV format) as input and converts it
        into an object whose attributes are used by algorithms to extract GPs.

        It provides the following attributes:
            thd_supp: minimum support threshold

            equal: eq value

            titles: column names of data source

            data: all the objects organized into their respective column

            row_count: number of objects

            col_count: number of all columns

            time_cols: column indices of the columns with data-time objects

            attr_cols: column indices of the columns with numeric values

            valid_bins: valid bitmaps (in the form of ndarray) of all gradual items corresponding to the attr_cols,
            a bitmap is valid if its computed support is equal or greater than the minimum support threshold

            net_wins: a net-wins matrix constructed from valid gradual item bitmaps

            no_bins: True if all none of the attr_cols yields a valid bitmap

            gradual_patterns: list of GP objects

        >>> import so4gp as sgp
        >>> import pandas
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> columns = ['Age', 'Salary', 'Cars', 'Expenses']
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> data_gp = sgp.DataGP(data_source=dummy_df, min_sup=0.5)
        >>> data_gp.fit_bitmap()


        :param data_source: [required] a data source, it can either be a 'file in csv format' or a 'Pandas DataFrame'
        :type data_source: pd.DataFrame | str

        :param min_sup: [optional] minimum support threshold, the default is 0.5

        :param eq: encode equal values as gradual, the default is False

        """
        self.thd_supp = min_sup
        """:type thd_supp: float"""
        self.equal = eq
        """:type eq: bool"""
        self.titles, self.data = DataGP.read(data_source)
        """:type titles: list"""
        """:type data: np.ndarray"""
        self.row_count, self.col_count = self.data.shape
        self.time_cols = self._get_time_cols()
        self.attr_cols = self._get_attr_cols()
        self.valid_bins = np.array([])
        self.valid_tids = defaultdict(set)
        self.no_bins = False
        self.step_name = ''  # For T-GRAANK
        self.attr_size = 0  # For T-GRAANK
        self.gradual_patterns = None

    def _get_attr_cols(self):
        """Description

        Returns indices of all columns with non-datetime objects

        :return: ndarray
        """
        all_cols = np.arange(self.col_count)
        attr_cols = np.setdiff1d(all_cols, self.time_cols)
        return attr_cols

    def _get_time_cols(self):
        """Description

        Tests each column's objects for date-time values. Returns indices of all columns with date-time objects

        :return: ndarray
        """
        # Retrieve first column only
        time_cols = list()
        n = self.col_count
        for i in range(n):  # check every column/attribute for time format
            row_data = str(self.data[0][i])
            try:
                time_ok, t_stamp = DataGP.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        return np.array(time_cols)

    def get_gi_bitmap(self, col):
        """Description

        Computes and returns the bitmap matrix corresponding to an attribute.

        :param col: specific attribute (or column)
        :return: numpy (bitmap)
        """
        if col in self.time_cols:
            raise Exception("Error: " + str(self.titles[col][1].decode()) + " is a date/time column!")
        elif col >= self.col_count:
            raise Exception("Error: Column does not exist!")
        else:
            attr_data = self.data.T
            # n = d_set.row_count
            col_data = np.array(attr_data[col], dtype=float)
            with np.errstate(invalid='ignore'):
                temp_pos = np.where(col_data < col_data[:, np.newaxis], 1, 0)
            return temp_pos

    def fit_bitmap(self, attr_data=None):
        """Description

        Generates bitmaps for columns with numeric objects. It stores the bitmaps in attribute valid_bins (those bitmaps
        whose computed support values are greater or equal to the minimum support threshold value).

        :param attr_data: stepped attribute objects
        :type attr_data: np.ndarray
        :return: void
        """
        # (check) implement parallel multiprocessing
        # 1. Transpose csv array data
        if attr_data is None:
            attr_data = self.data.T
            self.attr_size = self.row_count
        else:
            self.attr_size = len(attr_data[self.attr_cols[0]])

        # 2. Construct and store 1-item_set valid bins
        # execute binary rank to calculate support of pattern
        n = self.attr_size
        valid_bins = list()
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = np.array((col, '+'), dtype='i, S1')
            decr = np.array((col, '-'), dtype='i, S1')

            # 2a. Generate 1-itemset gradual items
            with np.errstate(invalid='ignore'):
                if not self.equal:
                    temp_pos = col_data > col_data[:, np.newaxis]
                else:
                    temp_pos = col_data >= col_data[:, np.newaxis]
                    np.fill_diagonal(temp_pos, 0)

                # 2b. Check support of each generated itemset
                supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)
                if supp >= self.thd_supp:
                    valid_bins.append(np.array([incr.tolist(), temp_pos], dtype=object))
                    valid_bins.append(np.array([decr.tolist(), temp_pos.T], dtype=object))
        self.valid_bins = np.array(valid_bins)
        # print(self.valid_bins)
        if len(self.valid_bins) < 3:
            self.no_bins = True
        gc.collect()

    def fit_tids(self):
        """Description

        Generates transaction ids (tids) for each column/feature with numeric objects. It stores the tids in attribute
        valid_tids (those tids whose computed support values are greater or equal to the minimum support threshold
        value).

        :return: void
        """
        self.fit_bitmap()
        n = self.row_count
        for bin_obj in self.valid_bins:
            arr_ij = np.transpose(np.nonzero(bin_obj[1]))
            set_ij = {tuple(ij) for ij in arr_ij if ij[0] < ij[1]}
            int_gi = int(bin_obj[0][0]+1) if (bin_obj[0][1].decode() == '+') else (-1 * int(bin_obj[0][0]+1))
            tids_len = len(set_ij)

            supp = float((tids_len*0.5) * (tids_len - 1)) / float(n * (n - 1.0) / 2.0)
            if supp >= self.thd_supp:
                self.valid_tids[int_gi] = set_ij

    @staticmethod
    def read(data_src):
        """Description

        Reads all the contents of a file (in CSV format) or a data-frame. Checks if its columns have numeric values. It
        separates its columns headers (titles) from the objects.

        :param data_src:
        :type data_src: pd.DataFrame | str

        :return: title, column objects
        """
        # 1. Retrieve data set from source
        if isinstance(data_src, pd.DataFrame):
            # a. DataFrame source
            # Check column names
            try:
                # Check data type
                _ = data_src.columns.astype(float)

                # Add column values
                data_src.loc[-1] = data_src.columns.to_numpy(dtype=float)  # adding a row
                data_src.index = data_src.index + 1  # shifting index
                data_src.sort_index(inplace=True)

                # Rename column names
                vals = ['col_' + str(k) for k in np.arange(data_src.shape[1])]
                data_src.columns = vals
            except ValueError:
                pass
            except TypeError:
                pass
            # print("Data fetched from DataFrame")
            return DataGP.clean_data(data_src)
        else:
            # b. CSV file
            file = str(data_src)
            try:
                with open(file, 'r') as f:
                    dialect = csv.Sniffer().sniff(f.readline(), delimiters=";,' '\t")
                    f.seek(0)
                    reader = csv.reader(f, dialect)
                    raw_data = list(reader)
                    f.close()

                if len(raw_data) <= 1:
                    raise Exception("CSV file read error. File has little or no data")
                else:
                    # print("Data fetched from CSV file")
                    # 2. Get table headers
                    keys = np.arange(len(raw_data[0]))
                    if raw_data[0][0].replace('.', '', 1).isdigit() or raw_data[0][0].isdigit():
                        vals = ['col_' + str(k) for k in keys]
                        header = np.array(vals, dtype='S')
                    else:
                        if raw_data[0][1].replace('.', '', 1).isdigit() or raw_data[0][1].isdigit():
                            vals = ['col_' + str(k) for k in keys]
                            header = np.array(vals, dtype='S')
                        else:
                            header = np.array(raw_data[0], dtype='S')
                            raw_data = np.delete(raw_data, 0, 0)
                    d_frame = pd.DataFrame(raw_data, columns=header)
                    return DataGP.clean_data(d_frame)
            except Exception as error:
                raise Exception("Error: " + str(error))

    @staticmethod
    def test_time(date_str):
        """Description

        Tests if a str represents a date-time variable.

        :param date_str: str value
        :type date_str: str
        :return: bool (True if it is date-time variable, False otherwise)
        """
        # add all the possible formats
        try:
            if type(int(date_str)):
                return False, False
        except ValueError:
            try:
                if type(float(date_str)):
                    return False, False
            except ValueError:
                try:
                    date_time = parse(date_str)
                    t_stamp = time.mktime(date_time.timetuple())
                    return True, t_stamp
                except ValueError:
                    raise ValueError('no valid date-time format found')

    @staticmethod
    def clean_data(df):
        """Description

        Cleans a data-frame (i.e., missing values, outliers) before extraction of GPs
        :param df: data-frame
        :type df: pd.DataFrame
        :return: list (column titles), numpy (cleaned data)
        """
        # 1. Remove objects with Null values
        df = df.dropna()

        # 2. Remove columns with Strings
        cols_to_remove = []
        for col in df.columns:
            try:
                _ = df[col].astype(float)
            except ValueError:
                # Keep time columns
                try:
                    ok, stamp = DataGP.test_time(str(df[col][0]))
                    if not ok:
                        cols_to_remove.append(col)
                except ValueError:
                    cols_to_remove.append(col)
                pass
            except TypeError:
                cols_to_remove.append(col)
                pass
        # keep only the columns in df that do not contain string
        df = df[[col for col in df.columns if col not in cols_to_remove]]

        # 3. Return titles and data
        if df.empty:
            raise Exception("Data set is empty after cleaning.")

        keys = np.arange(df.shape[1])
        values = np.array(df.columns, dtype='S')
        titles = list(np.rec.fromarrays((keys, values), names=('key', 'value')))
        # print("Data cleaned")
        # print(type(titles))
        return titles, df.values
