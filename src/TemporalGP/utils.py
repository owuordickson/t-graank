# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0
# See the LICENSE file in the root of this
# repository for complete details.



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



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



def classify_ftgps(lst_test_data, lst_ground_truth,threshold=0.5) -> dict:
    """
    Classify FTGPs by comparing a test dataset against a reference dataset.

    Each FTGP is classified according to whether its support is above or
    below the specified threshold in the test and reference datasets.

    A pattern that is absent from one dataset is assigned support = 0
    in that dataset.

    Pattern matching is performed using ``is_similar_to()`` and each
    pattern is matched at most once.

    Parameters
    ----------
    lst_test_data : list
        FTGPs extracted from the test dataset.

    lst_ground_truth : list
        FTGPs extracted from the reference (proxy ground-truth) dataset.

    threshold : float, default=0.5
        Support threshold used to distinguish supported and unsupported
        FTGPs.

    Returns
    -------
    dict
        Counts of TP, FP, FN, and TN.
    """

    confusion_matrix_counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    # Track which patterns have already been matched.
    matched_test = set()
    matched_ground_truth = set()

    # ------------------------------------------------------------
    # Step 1: Match test FTGPs to reference FTGPs.
    # ------------------------------------------------------------
    for i, test_pat in enumerate(lst_test_data):

        for j, gt_pat in enumerate(lst_ground_truth):

            if i in matched_test or j in matched_ground_truth:
                continue

            if test_pat.is_similar_to(gt_pat):

                test_support = test_pat.support
                gt_support = gt_pat.support

                if test_support >= threshold and gt_support >= threshold:
                    confusion_matrix_counts["TP"] += 1

                elif test_support >= threshold > gt_support:
                    confusion_matrix_counts["FP"] += 1

                elif test_support < threshold <= gt_support:
                    confusion_matrix_counts["FN"] += 1

                else:
                    confusion_matrix_counts["TN"] += 1

                matched_test.add(i)
                matched_ground_truth.add(j)

                break

    # ------------------------------------------------------------
    # Step 2: Handle test FTGPs with no corresponding reference FTGP.
    # Their reference support is therefore zero.
    # ------------------------------------------------------------
    for i, test_pat in enumerate(lst_test_data):

        if i in matched_test:
            continue

        if test_pat.support >= threshold:
            confusion_matrix_counts["FP"] += 1
        else:
            confusion_matrix_counts["TN"] += 1

    # ------------------------------------------------------------
    # Step 3: Handle reference FTGPs with no corresponding test FTGP.
    # Their test support is therefore zero.
    # ------------------------------------------------------------
    for j, gt_pat in enumerate(lst_ground_truth):

        if j in matched_ground_truth:
            continue

        if gt_pat.support >= threshold:
            confusion_matrix_counts["FN"] += 1
        else:
            confusion_matrix_counts["TN"] += 1

    return confusion_matrix_counts
