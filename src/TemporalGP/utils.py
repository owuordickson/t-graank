# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0
# See the LICENSE file in the root of this
# repository for complete details.



import numpy as np
from dtw import dtw

def gen_dtw_data(t_grad_obj, orig_rain_data, orig_evi_data, trans_rain_data, trans_evi_data):

    tgt_col = t_grad_obj.target_col
    target_col = t_grad_obj.titles[tgt_col][:3]

    dtw_data = {f"Rain ({target_col})": {}, f"Transformed Rain ({target_col})": {}, f"EVI ({target_col})": {},
                f"Transformed EVI ({target_col})": {}}

    target = orig_rain_data[1:, tgt_col].astype(float)
    for i, col in enumerate(t_grad_obj.titles):
        if i in t_grad_obj.time_cols:
            continue
        if i == tgt_col:
            pass
        else:
            query = orig_rain_data[1:, i].astype(float)
            alignment = dtw(query, target, keep_internals=True) if target is not None else None
            if alignment is not None:
                dtw_data[f"Rain ({target_col})"].update({col: float(alignment.distance)})

    target = trans_rain_data[1:, tgt_col].astype(float)
    for i, col in enumerate(t_grad_obj.titles):
        if i in t_grad_obj.time_cols:
            continue
        if i == tgt_col:
            pass
        else:
            query = trans_rain_data[1:, i].astype(float)
            alignment = dtw(query, target, keep_internals=True) if target is not None else None
            if alignment is not None:
                dtw_data[f"Transformed Rain ({target_col})"].update({col: float(alignment.distance)})

    target = orig_evi_data[1:, tgt_col].astype(float)
    for i, col in enumerate(t_grad_obj.titles):
        if i in t_grad_obj.time_cols:
            continue
        if i == tgt_col:
            pass
        else:
            query = orig_evi_data[1:, i].astype(float)
            alignment = dtw(query, target, keep_internals=True) if target is not None else None
            if alignment is not None:
                dtw_data[f"EVI ({target_col})"].update({col: float(alignment.distance)})

    target = trans_evi_data[1:, tgt_col].astype(float)
    for i, col in enumerate(t_grad_obj.titles):
        if i in t_grad_obj.time_cols:
            continue
        if i == tgt_col:
            pass
        else:
            query = trans_evi_data[1:, i].astype(float)
            alignment = dtw(query, target, keep_internals=True) if target is not None else None
            if alignment is not None:
                dtw_data[f"Transformed EVI ({target_col})"].update({col: float(alignment.distance)})
    return dtw_data


def gen_euc_data(t_grad_obj, orig_rain_data, orig_evi_data, trans_rain_data, trans_evi_data):

    tgt_col = t_grad_obj.target_col
    target_col = t_grad_obj.titles[tgt_col][:3]

    euc_data = {f"Rain ({target_col})": {}, f"Transformed Rain ({target_col})": {}, f"EVI ({target_col})": {},
                f"Transformed EVI ({target_col})": {}}

    target = orig_rain_data[1:, tgt_col].astype(float)
    for i, col in enumerate(t_grad_obj.titles):
        if i in t_grad_obj.time_cols:
            continue
        if i == tgt_col:
            pass
        else:
            query = orig_rain_data[1:, i].astype(float)
            distance = np.linalg.norm(target - query) if target is not None else None
            if distance is not None:
                euc_data[f"Rain ({target_col})"].update({col: float(distance)})

    target = trans_rain_data[1:, tgt_col].astype(float)
    for i, col in enumerate(t_grad_obj.titles):
        if i in t_grad_obj.time_cols:
            continue
        if i == tgt_col:
            pass
        else:
            query = trans_rain_data[1:, i].astype(float)
            distance = np.linalg.norm(target - query) if target is not None else None
            if distance is not None:
                euc_data[f"Transformed Rain ({target_col})"].update({col: float(distance)})

    target = orig_evi_data[1:, tgt_col].astype(float)
    for i, col in enumerate(t_grad_obj.titles):
        if i in t_grad_obj.time_cols:
            continue
        if i == tgt_col:
            pass
        else:
            query = orig_evi_data[1:, i].astype(float)
            distance = np.linalg.norm(target - query) if target is not None else None
            if distance is not None:
                euc_data[f"EVI ({target_col})"].update({col: float(distance)})

    target = trans_evi_data[1:, tgt_col].astype(float)
    for i, col in enumerate(t_grad_obj.titles):
        if i in t_grad_obj.time_cols:
            continue
        if i == tgt_col:
            pass
        else:
            query = trans_evi_data[1:, i].astype(float)
            distance = np.linalg.norm(target - query) if target is not None else None
            if distance is not None:
                euc_data[f"Transformed EVI ({target_col})"].update({col: float(distance)})
    return euc_data
