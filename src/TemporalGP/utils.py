# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0
# See the LICENSE file in the root of this
# repository for complete details.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtw import dtw


def compute_distance(t_grad_obj, orig_rain_data, orig_evi_data, trans_rain_data, trans_evi_data):


    tgt_col = t_grad_obj.target_col
    target_col = t_grad_obj.titles[tgt_col][:3]
    dtw_data = {f"Orig ({target_col})": {}, f"Transformed ({target_col})": {}}
    euc_data = {f"Orig ({target_col})": {}, f"Transformed ({target_col})": {}}

    def compute_dtw_distance(arr_data1, arr_data2, str_key: str = ""):
        """
        Compute DTW alignment distance between the target column in one time-series and all other columns in another time-series.
        """
        target = arr_data1[1:, tgt_col].astype(float)
        for i, col in enumerate(t_grad_obj.titles):
            if i in t_grad_obj.time_cols:
                continue
            if i == tgt_col:
                pass
            else:
                query = arr_data2[1:, i].astype(float)
                alignment = dtw(query, target, keep_internals=True) if target is not None else None
                if alignment is not None:
                    align_dist = getattr(alignment, 'distance', np.inf) if alignment else np.inf
                    dtw_data[str_key].update({col: float(align_dist)})

    def compute_euclidean_distance(arr_data1, arr_data2, str_key: str = ""):
        """
        Compute Euclidean distance between the target column in one time-series and all other columns in another time-series.
        """
        target = arr_data1[1:, tgt_col].astype(float)
        for i, col in enumerate(t_grad_obj.titles):
            if i in t_grad_obj.time_cols:
                continue
            if i == tgt_col:
                pass
            else:
                query = arr_data2[1:, i].astype(float)
                distance = np.linalg.norm(target - query) if target is not None else None
                if distance is not None:
                    euc_data[str_key].update({col: float(distance)})

    compute_dtw_distance(orig_rain_data, orig_evi_data, str_key=f"Orig ({target_col})")
    compute_dtw_distance(trans_rain_data, trans_evi_data, str_key=f"Transformed ({target_col})")
    # compute_dtw_distance(orig_evi_data, str_key=f"EVI ({target_col})")
    # compute_dtw_distance(trans_evi_data, str_key=f"Transformed EVI ({target_col})")

    compute_euclidean_distance(orig_rain_data, orig_evi_data, str_key=f"Orig ({target_col})")
    compute_euclidean_distance(trans_rain_data, trans_evi_data, str_key=f"Transformed ({target_col})")
    # compute_euclidean_distance(orig_evi_data, str_key=f"EVI ({target_col})")
    # compute_euclidean_distance(trans_evi_data, str_key=f"Transformed EVI ({target_col})")

    return dtw_data, euc_data



def gp_descriptor_spider_plot(df_list: list[pd.DataFrame], labels: list[str], parameters: list[str], grid_levels: int = 6) -> plt.Figure:
    """
    Modified to accept a list of DataFrames (e.g., [stats_rain, stats_evi])
    and plot their mean/std comparisons.

    :param df_list: A list of DataFrames containing GP descriptors as floats.
    :param labels: A list of labels corresponding to each DataFrame.
    :param parameters: A list of parameter names (or columns of the DataFrames) to plot.
    :param grid_levels: The number of levels in the hexagon grid.

    :return: A matplotlib figure object containing the spider plot.
    """

    # --- Data Processing ---
    df_avgs = []
    df_stds = []
    # Extract means and stds for each dataframe provided
    for df in df_list:
        # Filter only for the requested parameters to ensure order
        df_avgs.append(df[parameters].mean())
        df_stds.append(df[parameters].std())

    # --- Helper Functions (Local) ---
    def format_scale_value(value):
        if abs(value) >= 1_000: return f'{value / 1_000:.1f}K'
        elif abs(value) >= 10: return f'{value:.1f}'
        else: return f'{value:.2f}'

    def shift_value(val):
        return val - min_val

    def compute_grid_scale(min_v, max_v):
        # Create dynamic levels from min to max
        return np.linspace(min_v, max_v, grid_levels)

    # --- Plot Setup ---
    num_vars = len(parameters)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    fig = plt.figure(figsize=(11, 8.5), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    # Determine global max/min for scaling
    all_highs = [avg + std for avg, std in zip(df_avgs, df_stds)]
    all_lows = [avg - std for avg, std in zip(df_avgs, df_stds)]
    max_val = max([v.max() for v in all_highs])
    min_val = min([v.min() for v in all_lows])

    levels = compute_grid_scale(min_val, max_val)
    max_shifted = shift_value(max(levels))

    # --- Draw Hexagon Grid ---
    for level in levels:
        shifted_level = shift_value(level)
        x_grid = shifted_level * np.cos(np.append(angles, angles[0]))
        y_grid = shifted_level * np.sin(np.append(angles, angles[0]))
        ax.plot(x_grid, y_grid, 'k-', linewidth=0.5, alpha=0.2)
        ax.text(shifted_level, 0.05, format_scale_value(level), ha='left', va='bottom', fontsize=7, alpha=0.4)

    # Draw spokes
    for angle in angles:
        ax.plot([0, max_shifted * np.cos(angle)], [0, max_shifted * np.sin(angle)], 'k-', linewidth=0.5, alpha=0.3)

    # --- Plot Data Groups ---
    for i, (avg, std) in enumerate(zip(df_avgs, df_stds)):
        values = shift_value(avg.values)
        errors = std.values

        # Cartesian conversion
        x = values * np.cos(angles)
        y = values * np.sin(angles)

        # Plot Polygon
        poly_line, = ax.plot(np.append(x, x[0]), np.append(y, y[0]), linewidth=2, label=labels[i])
        color = poly_line.get_color()
        ax.fill(x, y, alpha=0.1, color=color)

        # Perpendicular Error Bars
        dx_perp = -np.sin(angles)
        dy_perp = np.cos(angles)
        err_scale = 1.5 # Adjusted for standard deviation visibility

        for j in range(len(angles)):
            xi, yi = x[j], y[j]
            err = errors[j] * err_scale
            ax.plot([xi + err * dx_perp[j], xi - err * dx_perp[j]],
                    [yi + err * dy_perp[j], yi - err * dy_perp[j]],
                    color=color, linewidth=1)

    # --- Final Touches ---
    label_dist = max_shifted * 1.1
    for i, (angle, param) in enumerate(zip(angles, parameters)):
        # Convert angle to degrees for matplotlib rotation
        angle_deg = np.rad2deg(angle)
        check_angle = np.round(angle_deg, 0) % 360

        # Rename long parameters for clarity
        param = param.replace('Avg. Deviation from Diagonal', 'Avg. Deviation')

        # Tilt Logic:
        # Flip text if it's on the left side (between 90 and 270 degrees)
        # to keep it right-side up.
        if check_angle == 0 or check_angle == 180:
            display_angle = 90
        elif check_angle == 120 or check_angle == 300:
            display_angle = 30
        elif check_angle == 240 or check_angle == 60:
            display_angle = -30
        else:
            display_angle = 0

        # Calculate position
        x_pos = label_dist * np.cos(angle)
        y_pos = label_dist * np.sin(angle)

        ax.text(
            x_pos, y_pos, param,
            ha='center', va='center',
            fontsize=10,
            fontweight='bold',
            rotation=display_angle,      # Apply the tilt
            rotation_mode='anchor'       # Ensures rotation is around the text center
        )

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("GP Descriptors Spider Plot with Std. Dev. Error Bars", fontsize=14, pad=40)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

    return fig



def classify_ftgps(lst_test_data, lst_ground_truth) -> dict:
    """
    Classify extracted FTGPS into TP, FP, FN, TN.

    :param lst_test_data: List of extracted FTGPS from the test data.
    :param lst_ground_truth: List of ground truth FTGPS.

    :return: Dictionary containing counts of TP, FP, FN, TN.
    """

    # def exists_in(pat, lst):
    #    for pat_i in lst:
    #        if pat.is_similar_to(pat_i):
    #            return True
    #    return False

    res_cat_count = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    total = 0
    already_seen = set()
    for pat1 in lst_test_data:
        for pat2 in lst_ground_truth:
            if pat1.is_similar_to(pat2) and pat2 not in already_seen:
                already_seen.add(pat2)
                if pat1.support >= 0.5 and pat2.support >= 0.5:
                    res_cat_count["TP"] += 1
                    total += 1
                elif pat1.support >= 0.5 > pat2.support:
                    res_cat_count["FP"] += 1
                    total += 1
                elif pat1.support < 0.5 <= pat2.support:
                    res_cat_count["FN"] += 1
                    total += 1
                elif pat1.support < 0.5 > pat2.support:
                    res_cat_count["TN"] += 1
                    total += 1
    missing = len(lst_test_data) - total
    res_cat_count["FP"] += missing
    return res_cat_count



def gen_distance_plot(lst_tgrad_objs, dtw_data, euc_data) -> plt.Figure:
    """
    Generate distance plots for each time-series.

    :param lst_tgrad_objs: List of TGrad objects.
    :param dtw_data: Dictionary containing DTW distance data.
    :param euc_data: Dictionary containing Euclidean distance data.

    :return: A matplotlib figure object containing the distance plots.
    """
    # Create subplots
    fig, ax = plt.subplots(4, 2, figsize=(16, 20))

    for i, t_grad in enumerate(lst_tgrad_objs):
        locations = []
        for j in t_grad.feature_cols:
            col = t_grad.titles[j]
            col = col[:3]
            locations.append(col)
        target_col = t_grad.titles[t_grad.target_col][:3]
        x = np.arange(len(locations))
        width = 0.36

        columns = ['Data', 'Metric']
        columns.extend(locations)

        plt_data = []
        for key, val_dict in dtw_data[i].items():
            row = [key, "DTW"]
            for k, v in val_dict.items():
                row.append(v)
            plt_data.append(row)

        for key, val_dict in euc_data[i].items():
            row = [key, "EUC"]
            for k, v in val_dict.items():
                row.append(v)
            plt_data.append(row)
        plt_df = pd.DataFrame(plt_data, columns=columns)

        # 2. Filter data for specific plots
        # df_evi = plt_df[plt_df['Data'].str.contains('EVI')]
        df_euc = plt_df[plt_df['Metric'] == 'EUC']
        df_dtw = plt_df[plt_df['Metric'] == 'DTW']

        dtw_rain_vals = df_dtw[df_dtw['Data'].str.contains('Orig')][locations].values.flatten()
        dtw_trans_vals = df_dtw[df_dtw['Data'].str.contains('Transformed')][locations].values.flatten()
        euc_rain_vals = df_euc[df_euc['Data'].str.contains('Orig')][locations].values.flatten()
        euc_trans_vals = df_euc[df_euc['Data'].str.contains('Transformed')][locations].values.flatten()

        ## Plot 1: All EUC Data
        colors = ['#4CAF50', '#F44336', '#FF9800', '#2196F3']  # Green, Red, Orange, Blue
        ax[i, 0].bar(x - width / 2, euc_rain_vals, width, label='Original Data', color=colors[1])
        ax[i, 0].bar(x + width / 2, euc_trans_vals, width, label='Transformed Data', color=colors[2])
        ax[i, 0].set_title('Euclidean Distance (EVI vs Rain Data)')
        ax[i, 0].set_ylabel(f'{target_col}')
        # ax[i,0].set_yscale('log') # Use log scale because Rain is much higher than EVI
        ax[i, 0].set_xticks(x)
        ax[i, 0].set_xticklabels(locations)
        ax[i, 0].legend(loc='upper right')
        ax[i, 0].grid(True, alpha=0.3)
        ax[i, 0].set_xlabel('Location')

        ## Plot 2: All DTW Metric Data
        # for _, row in df_dtw.iterrows():
        # lbl_txt = str(row['Legend']).split('(')[0] + ' Data'
        # ax[i,1].plot(locations, row[locations], marker='o', label=lbl_txt)
        #    ax[i,1].bar(locations, row[locations], color=colors)#, label=lbl_txt)
        ax[i, 1].bar(x - width / 2, dtw_rain_vals, width, label='Original Data', color=colors[0])
        ax[i, 1].bar(x + width / 2, dtw_trans_vals, width, label='Transformed Data', color=colors[3])
        ax[i, 1].set_title('DTW Alignment Distance (EVI vs Rain Data)')
        ax[i, 1].set_ylabel(f'{target_col}')
        # ax[i,1].set_yscale('log') # Use log scale because Rain is much higher than EVI
        ax[i, 1].set_xticks(x)
        ax[i, 1].set_xticklabels(locations)
        ax[i, 1].legend(loc='upper right')
        ax[i, 1].grid(True, which="both", ls="-", alpha=0.2)
        ax[i, 1].set_xlabel('Location')

    fig.tight_layout()
    return fig
