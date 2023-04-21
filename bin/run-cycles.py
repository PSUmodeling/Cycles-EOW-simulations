#!/usr/bin/env python3

import argparse
import csv
import numpy as np
import os
import pandas as pd
import subprocess
from datetime import datetime
from math import inf
from string import Template

"""Run Cycles simulations for different crops under different nuclear war scenarios

Run Cycles simulations
"""

CYCLES = './bin/Cycles'
BASE_TMP = 6.0
SPIN_UP = True
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
LOOKUP = lambda lut, crop: f'./data/{crop}_rainfed_{lut.lower()}_lookup_3.1.csv'
RUNS = lambda lut, scenario, crop: f'./data/{lut.lower()}_{scenario}_{crop}_runs.csv' if lut == 'EOW' else f'./data/{lut.lower()}_{crop}_runs.csv'
SUMMARY = lambda lut, scenario, crop: f'summary/{lut.lower()}_{scenario}_{crop}.csv' if lut == 'EOW' else f'summary/{lut.lower()}_{crop}.csv'

RM_CYCLES_IO = 'rm -fr input/*.ctrl input/*.soil output/*'
RM_OPERATION = 'rm -f input/*.operation'

def calculate_months_for_planting(weather, tmp_max, tmp_min):
    """Calculate months in which crops can be planted
    """
    tmp_max = float(tmp_max) if tmp_max != '-999' else 100.0    # -999 is the special case that disables max temperature
    tmp_min = float(tmp_min)

    # Read weather file with comment lines removed
    cols = {
        'YEAR': int,
        'DOY': int,
        'PP': float,
        'TX': float,
        'TN': float,
        'SOLAR': float,
        'RHX': float,
        'RHN': float,
        'WIND': float,
    }
    df = pd.read_csv(
        f'input/weather/{weather}',
        names=cols.keys(),
        comment='#',
        delim_whitespace=True,
        na_values=[-999],
    )
    df = df.iloc[4:, :]
    df = df.astype(cols)

    # Calculate month, average temperature, 7-day moving average temperature, thermal time
    df['tavg'] = 0.5 * df['TX'] + 0.5 * df['TN']
    df['tt'] = df.apply(lambda x: 0.0 if x['tavg'] < BASE_TMP else x['tavg'] - BASE_TMP, axis=1)
    df['tma'] = df.rolling(7, center=True, min_periods=1).mean()['tavg']

    # Calculate average thermal time
    tt = df['tt'].sum() / (len(df) / 365.0)

    # Filter out days outside allowed temperature range
    df = df[(df['tma'] > tmp_min) & (df['tma'] < tmp_max)]
    if df.empty:
        months = []
    else:
        df['month'] = df.apply(lambda x: datetime.strptime('2009-' + '%d' %(x['DOY']), '%Y-%j').strftime('%m'), axis=1)
        months = df['month'].unique()

    # Return a list months that contain days inside temperature range
    return months, tt


def generate_cycles_input(gid, crop, soil, weather, tmp_max, tmp_min, start_year, end_year, month):
    with open(f'data/template.operation') as op_file:
        op_src = Template(op_file.read())

    with open(f'data/template.ctrl') as ctrl_file:
        ctrl_src = Template(ctrl_file.read())

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
        'start': start_year,
        'end': end_year,
        'operation': f'M{month}.operation',
        'soil': f'soil/{soil}',
        'weather': f'weather/{weather}',
    }
    result = ctrl_src.substitute(ctrl_data)
    with open(f'./input/{gid}_M{month}.ctrl', 'w') as f:
        f.write(result + '\n')


def run_cycles(spin_up, simulation):
    cmd = [
        CYCLES,
        '-sb' if spin_up else '-b',
        simulation,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return result.returncode


def find_optimal_planting_dates(gid, end_year, months):
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
            df = df[df['year'] != end_year]
            yield_avg[int(month) - 1] = df['grain_yield'].sum()
        except:
            continue

    # If there is no yield from any month, return 0
    if not any(yield_avg):
        return 0

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

    return ref_month


def find_ref_month_crop(gid, crop, soil, weather, tmp_max, tmp_min, start_year, end_year):
    if not os.path.exists(f'input/weather/{weather}'):
        print(f'Weather file error')
        return np.nan, ''

    if not os.path.exists(f'input/soil/{soil}'):
        print(f'Soil file error')
        return np.nan, ''

    ## Find which months are potentially suitable to plant crops. Calculate thermal times for choosing RM
    months, tt = calculate_months_for_planting(weather, tmp_max, tmp_min)

    if len(months) == 0:
        print(f'Unsuitable climate')
        return np.nan, ''

    # Find which RM to be planted
    ## Find the RM type with closest thermal time
    ## https://stackoverflow.com/questions/52844099/finding-the-closest-value-in-a-python-dictionary-and-returning-its-key
    crop_rm, _ = min(MATURITY_TTS[crop].items(), key=lambda x: abs(tt * 0.85 - x[1]))
    ## Find the RM type with closest thermal time that is greater than the target thermal time
    ## https://stackoverflow.com/questions/68594435/find-the-number-closest-to-and-greater-than-an-input-number-in-a-dictionary
    #crop_rm, _ = min(MATURITY_TTS[crop].items(), key=lambda x: tt * 0.85 - x[1] if tt * 0.85 - x[1] > 0 else inf)

    ## Run each month
    for month in months:
        generate_cycles_input(gid, crop_rm, soil, weather, tmp_max, tmp_min, start_year, end_year, month)

        ### Run Cycles
        if run_cycles(not SPIN_UP, f'{gid}_M{month}') != 0:
            print(f'Cycles error')
            return np.nan, ''
    else:
        ## Read season files and find optimal planting months
        ref_month = find_optimal_planting_dates(gid, end_year, months)

        if ref_month <= 0:
            print(f'No yield')
            return np.nan, ''

    return ref_month, crop_rm


def main(params):

    pre_run = params['pre_run']
    lut = params['lut']
    scenario = params['scenario']
    crop = params['crop']
    start_year = '%4.4d' % params['start']
    end_year = '%4.4d' % params['end']

    os.makedirs('summary', exist_ok=True)

    tmp_max = MAX_TMPS[crop]
    tmp_min = MIN_TMPS[crop]

    fn = RUNS(lut, scenario, crop) if pre_run else SUMMARY(lut, scenario, crop)

    # Read in look-up table or run table
    with open(LOOKUP(lut, crop) if pre_run else RUNS(lut, scenario, crop)) as f:
        reader = csv.reader(f, delimiter=',')

        headers = next(reader)
        data = [{h:x for (h,x) in zip(headers,row)} for row in reader]

    first = True

    with open(fn, 'w') as output_fp:
        # Run each region
        for row in data:
            if not row: continue    # Skip empty lines

            gid = row['GID']
            weather = f'{scenario}_{row["Weather"]}.weather' if lut == 'EOW' else row['Weather']
            soil = row['Soil']

            print(
                f'{gid} - [{weather}, {soil}] - ',
                end=''
            )

            if pre_run:
                ref_month, crop_rm = find_ref_month_crop(gid, crop, soil, weather, tmp_max, tmp_min, start_year, end_year)

                if not np.isnan(ref_month):
                    print('Success')

                    if first:
                        output_fp.write(','.join(['GID', 'Weather', 'Soil', 'AreaKm2', 'AreaFraction', 'RefMonth', 'Crop']))
                        output_fp.write('\n')
                        first = False

                    strs = [gid, row['Weather'], soil, row['AreaKm2'], row['AreaFraction'], str(ref_month), crop_rm]
                    output_fp.write(','.join(strs))
                    output_fp.write('\n')
            else:
                ref_month = '%2.2d' % (int(row['RefMonth']))
                crop_rm = row['Crop']

                # Run Cycles again with spin-up
                generate_cycles_input(gid, crop_rm, soil, weather, tmp_max, tmp_min, start_year, end_year, ref_month)
                run_cycles(SPIN_UP, f'{gid}_M{ref_month}')

                # Return season file with best yield
                df = pd.read_csv(
                    f'output/{gid}_M{ref_month}/season.txt',
                    sep='\t',
                    header=0,
                    skiprows=[1],
                    skipinitialspace=True,
                )
                df = df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))
                df['crop'] = df['crop'].str.strip()
                df.insert(0, 'gid', gid)
                df.insert(1, 'area_km2', row['AreaKm2'])
                df.insert(2, 'area_fraction', row['AreaFraction'])

                print('Success')

                if first:
                    strs = df.to_csv(index=False)
                    first = False
                else:
                    strs = df.to_csv(header=False, index=False)

                output_fp.write(''.join(strs))

            ## Remove generated input/output files
            subprocess.run(
                RM_CYCLES_IO,
                shell='True',
            )

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
        '--lut',
        default='global',
        choices=['global', 'CONUS', 'EOW', 'test'],
        help='Look-up table to be used',
    )
    parser.add_argument(
        '--scenario',
        default='nw_cntrl_03',
        choices=SCENARIOS,
        help='EOW NW scenario',
    )
    parser.add_argument(
        '--start',
        required=True,
        type=int,
        help='Start year of simulation (use 0005 for EOW simulations)',
    )
    parser.add_argument(
        '--end',
        required=True,
        type=int,
        help='End year of simulation (use 0019 for EOW simulations)',
    )
    parser.add_argument(
        '--pre-run',
        default=False,
        action='store_true',
    )
    args = parser.parse_args()

    main(vars(args))


if __name__ == '__main__':
    _main()
