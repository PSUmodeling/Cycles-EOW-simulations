#!/usr/bin/env python3

import argparse
import csv
import numpy as np
import os
import pandas as pd
import subprocess
from datetime import datetime
from string import Template

"""Run Cycles simulations for different crops under different nuclear war scenarios

Run Cycles simulations
"""

START_YEAR = '0005'
END_YEAR = '0019'
CYCLES = './bin/Cycles'
BASE_TMP = 6.0
MAX_TMPS = {
    'maize': '-999',
    'springwheat': '-999',
    'winterwheat': '15.0',
}
MIN_TMPS = {
    'maize': '12.0',
    'springWheat': '5.0',
    'winterWheat': '-999',
}
CROPS = {
    'maize',
    'springwheat',
    'winterwheat',
}
MATURITY_TTS = {
    'maize': {
        'CornRM.115': 2425.0,
        'CornRM.110': 2300.0,
        'CornRM.105': 2175.0,
        'CornRM.100': 2050.0,
        'CornRM.95': 1925.0,
        'CornRM.90': 1800.0,
        'CornRM.85': 1675.0,
        'CornRM.80': 1550.0,
        'CornRM.75': 1425.0,
        'CornRM.70': 1300.0,
    }
}
DOYS = {
    '01': [1, 31],
    '02': [32, 59],
    '03': [60, 90],
    '04': [91, 120],
    '05': [121, 151],
    '06': [152, 181],
    '07': [182, 212],
    '08': [213, 243],
    '09': [244, 273],
    '10': [274, 304],
    '11': [305, 334],
    '12': [335, 365],
}
SCENARIOS = [
    'nw_cntrl_03',
    'nw_targets_01',
    'nw_targets_02',
    'nw_targets_03',
    'nw_targets_04',
    'nw_targets_05',
    'nw_ur_150_07',
]
LOOKUP = lambda crop: f'./data/{crop}_rainfed_eow_lookup_3.0.csv'

RM_CYCLES_IO = 'rm -fr input/*.ctrl input/*.soil output/*'
RM_OPERATION = 'rm -f input/*.operation'

def calculate_months_for_planting(weather, tmp_max, tmp_min):
    """Calculate months in which crops can be planted
    """
    tmp_max = float(tmp_max) if tmp_max != '-999' else 100.0    # -999 is the special case that disables max temperature
    tmp_min = float(tmp_min)

    # Read weather file with comment lines removed
    df = pd.read_csv(
        f'input/weather/{weather}',
        comment='#',
        skiprows=range(0, 4),
        delim_whitespace=True,
        na_values=[-999],
    )

    # Calculate month, average temperature, 7-day moving average temperature, thermal time
    df['tavg'] = 0.5 * df['TX'] + 0.5 * df['TN']
    df['tt'] = df.apply(lambda x: 0.0 if x['tavg'] < BASE_TMP else x['tavg'] - BASE_TMP, axis=1)
    df['tma'] = df.rolling(7, center=True, min_periods=1).mean()['tavg']

    # Calculate average thermal time
    tt = df['tt'].sum() / (len(df) / 365.0)

    # Filter out days outside allowed temperature range
    df = df[(df['tma'] > tmp_min) & (df['tma'] < tmp_max)]
    df['month'] = df.apply(lambda x: datetime.strptime('2009-' + '%d' %(x['DOY']), '%Y-%j').strftime('%m'), axis=1)

    # Return a list months that contain days inside temperature range
    return df['month'].unique(), tt


def find_optimal_planting_dates(gid, months):
    """Find optimal planting months/dates for each gid

    Read season files from 12 planting months and find the month with the best yield with a X-month moving window
    average
    """
    yield_avg = np.zeros(12)

    # Read season files for each month
    for month in months:
        ## Read season file
        try:
            df = pd.read_csv(
                f'output/{gid}_M{month}/season.txt',
                sep='\t',
                usecols=[2, 5],
                names=['plant_date', 'grain_yield'],
                skiprows=[0, 1],
                skipinitialspace=True,
            )
            ## Filter out the last year (when planting late in the year, crop may not be harvested which causes a bias
            ## towards early in the year)
            df['year'] = df['plant_date'].str[0:4]
            df = df[df['year'] != 'END_YEAR']
            yield_avg[int(month) - 1] = df['grain_yield'].sum()
        except:
            continue

    # If there is no yield from any month, return an empty dataframe
    if not any(yield_avg):
        return pd.DataFrame()

    # Find optimal planting months
    HALF_WINDOW = 2
    max_yield = -999
    ref_month = -999

    for month in months:
        month = int(month) - 1

        if yield_avg[month] <= 0.0:
            continue

        ## Calculate X-month moving average yield
        m = month - 12 if month >= 12 - HALF_WINDOW else month  # Adjust to avoid using indices larger than 11

        yield_ma = yield_avg[np.r_[m - HALF_WINDOW:m + HALF_WINDOW + 1]].mean()

        ## Find optimal month
        if yield_ma > max_yield:
            max_yield = yield_ma
            ref_month = month + 1
        elif yield_ma == max_yield and max_yield > -999:
            ref_month = month + 1 if yield_avg[m] > yield_avg[ref_month - 1] else ref_month

    # Run Cycles again with spin-up
    cmd = [
        CYCLES,
        '-bs',
        f'{gid}_M{"%2.2d" % (ref_month)}',
    ]
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Return season file with best yield
    df = pd.read_csv(
        'output/%s_M%2.2d/season.txt' % (gid, ref_month),
        sep='\t',
        header=0,
        skiprows=[1],
        skipinitialspace=True,
    )
    df = df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))
    df['crop'] = df['crop'].str.strip()
    df.insert(0, 'gid', gid)

    return df


def main(params):
    # Create input directories
    os.makedirs('input/soil', exist_ok=True)
    os.makedirs('input/weather', exist_ok=True)
    os.makedirs('summary', exist_ok=True)

    summary_fp = f'summary/{params["scenario"]}_{params["crop"]}.txt'

    first = True

    tmp_max = MAX_TMPS[params['crop']]
    tmp_min = MIN_TMPS[params['crop']]

    # Read in look-up table
    with open(LOOKUP(params['crop'])) as f:
        reader = csv.reader(f, delimiter=',')

        headers = next(reader)
        data = [{h:x for (h,x) in zip(headers,row)} for row in reader]

    summary_strs = []
    # Run each region
    for row in data:
        gid = row['GID']

        weather = f'{params["scenario"]}_{row["Weather"]}.weather'
        soil = row['Soil']

        print(
            f'{gid} - [{weather}, {soil}] - ',
            end=''
        )

        if not os.path.exists(f'input/weather/{weather}'):
            print(f'Weather file error')
            continue

        if not os.path.exists(f'input/soil/{soil}'):
            print(f'Soil file error')
            continue

        ## Find which months are potentially suitable to plant crops. Calculate thermal times for choosing RM
        months, tt = calculate_months_for_planting(weather, tmp_max, tmp_min)

        if len(months) == 0:
            print(f'Unsuitable climate')
            continue

        # Find which RM to be planted
        crop, _ = min(MATURITY_TTS[params['crop']].items(), key=lambda x: abs(tt * 0.85 - x[1]))

        # Create operation files
        with open(f'data/template.operation') as op_file:
            op_src = Template(op_file.read())

        with open(f'data/template.ctrl') as ctrl_file:
            ctrl_src = Template(ctrl_file.read())

        ## Run each month
        for month in months:
            op_data = {
                'doy_start': DOYS[month][0],
                'doy_end': DOYS[month][1],
                'max_tmp': tmp_max,
                'min_tmp': tmp_min,
                'crop': crop,
            }
            result = op_src.substitute(op_data)
            with open(f'./input/M{month}.operation', 'w') as f:
                f.write(result + '\n')

            ### Create control file
            ctrl_data = {
                'start': START_YEAR,
                'end': END_YEAR,
                'operation': f'M{month}.operation',
                'soil': f'soil/{soil}',
                'weather': f'weather/{weather}',
            }
            result = ctrl_src.substitute(ctrl_data)
            with open(f'./input/{gid}_M{month}.ctrl', 'w') as f:
                f.write(result + '\n')

            ### Run Cycles
            cmd = [
                CYCLES,
                '-b',
                f'{gid}_M{month}',
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                print(f'Cycles error')
                break
        else:
            ## Read season files and find optimal planting months
            exdf = find_optimal_planting_dates(gid, months)

            if exdf.empty:
                print(f'No yield')
                continue
            else:
                print('Success')

            if first:
                summary_strs.append(
                    exdf.to_csv(index=False)
                )
                first = False
            else:
                summary_strs.append(
                    exdf.to_csv(header=False, index=False)
                )

        ## Remove generated input/output files
        subprocess.run(
            RM_CYCLES_IO,
            shell='True',
        )

    with open(summary_fp, 'w') as f:
        f.write(''.join(summary_strs))

    ## Remove operation files
    subprocess.run(
        RM_OPERATION,
        shell='True',
    )


def _main():
    parser = argparse.ArgumentParser(description='Cycles execution for a crop')
    parser.add_argument(
        '--crop',
        default='maize',
        choices=CROPS,
        help='Crop to be simulated',
    )
    parser.add_argument(
        '--scenario',
        default='nw_cntrl_03',
        choices=SCENARIOS,
        help='NW scenario',
    )
    args = parser.parse_args()

    main(vars(args))


if __name__ == '__main__':
    _main()
