#!/usr/bin/env python3

import argparse
import numpy as np
import os
import pandas as pd
import re
import subprocess
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from string import Template

"""Run Cycles simulations for different crops under different nuclear war scenarios

Run Cycles simulations
"""

SEVEN_ZIP = "./7zz"
START_YEAR = "0005"
END_YEAR = "0019"
LUIDS = ["10", "11", "12", "20", "30", "40"]

max_tmps = {
    "Maize": "-999",
    "SpringWheat": "-999",
    "WinterWheat": "15.0",
}

min_tmps = {
    "Maize": "12.0",
    "SpringWheat": "5.0",
    "WinterWheat": "-999",
}

crops = {
    "Maize": "CornRM.90",
    "SpringWheat": "SpringWheat",
    "WinterWheat": "WinterWheat",
}

doys = {
    "1": [1, 31],
    "2": [32, 59],
    "3": [60, 90],
    "4": [91, 120],
    "5": [121, 151],
    "6": [152, 181],
    "7": [182, 212],
    "8": [213, 243],
    "9": [244, 273],
    "10": [274, 304],
    "11": [305, 334],
    "12": [335, 365],
}

def calculate_months_for_planting(weather, tmp_max, tmp_min):
    """Calculate months in which crops can be planted
    """
    tmp_max = float(tmp_max) if tmp_max != "-999" else 100  # -999 is the special case that disables max temperature
    tmp_min = float(tmp_min)

    # Read weather file with comment lines removed
    with open(f"input/weather/{weather}", "r") as fp:
        weather_str = [line.strip() for line in fp if line.strip() and line.strip()[0] != "#"]

    # Read weather file into dataframe
    df = pd.read_csv(StringIO("\n".join(weather_str[3:])),
        delim_whitespace=True,
        na_values=[-999])

    # Calculate month, average temperature, and 7-day moving average temperature
    df["month"] = df.apply(lambda x: datetime.strptime("2009-" + "%d" %(x["DOY"]), "%Y-%j").strftime("%m"), axis=1)
    df["tavg"] = 0.5 * df["TX"] + 0.5 * df["TN"]
    df["tma"] = df.rolling(7, center=True, min_periods=1).mean()["tavg"]

    # Filter out days outside allowed temperature range
    df = df[(df["tma"] > tmp_min) & (df["tma"] < tmp_max)]

    # Return a list months that contain days inside temperature range
    return df["month"].unique()


def convert_soil(soil):
    """Convert soil files in database to latest Cycles format
    """
    # Read original soil file
    with open(f"input/{soil}", "r") as fp:
        soil_str = [line.strip() for line in fp if line.strip() and line.strip()[0] != "#"]

    # Write new soil file
    with open(f"input/soil/{soil}", "w") as fp:
        fp.write(soil_str[0] + "\n")    # Curve number line
        fp.write(soil_str[1] + "\n")    # Slope line
        fp.write(soil_str[2] + "\n")    # Total number of layers line
        ## Header line
        tmp = soil_str[3].split()[0:8]
        tmp.append("SON")   # Add SON header
        tmp.extend(soil_str[3].split()[8:10])
        tmp.extend(["BYP_H", "BYP_V"])  # Add bypass headers
        fp.write("\t".join(tmp) + "\n")
        ## Layers
        for kline in range(4, len(soil_str)):
            tmp = soil_str[kline].split()[0:8]
            tmp.append("-999")  # Add SON
            tmp.extend(soil_str[kline].split()[8:10])
            tmp.extend(["0.0", "0.0"])  # Add bypass parameters
            fp.write("\t".join(tmp) + "\n")


def find_optimal_planting_dates(grid, months):
    """Find optimal planting months/dates for each grid
    Read season files from 12 planting months and find the month with the best yield with a X-month moving window
    average"""
    yield_avg = np.zeros(12)

    # Read season files for each month
    for month in months:
        ## Read season file
        try:
            df = pd.read_csv(
                f"output/{grid}_M{month}/season.txt",
                sep="\t",
                header=0,
                skiprows=[1],
                skipinitialspace=True,
            )
            df = df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))
            ## Filter out the last year (when planting late in the year, crop may not be harvested which causes a bias
            ## towards early in the year)
            df["year"] = df["plant_date"].str[0:4]
            df = df[df["year"] != "END_YEAR"]
            yield_avg[int(month) - 1] = df["grain_yield"].sum()
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

        ## Calculate X-month moving average yield
        m = month - 12 if month >= 12 - HALF_WINDOW else month  # Adjust to avoid using indices larger than 11

        yield_ma = yield_avg[np.r_[m - HALF_WINDOW:m + HALF_WINDOW + 1]].mean()

        ## Find optimal month
        if yield_ma > max_yield:
            max_yield = yield_ma
            ref_month = month + 1
        elif yield_ma == max_yield and max_yield > -999:
            ref_month = month + 1 if yield_avg[m] > yield_avg[ref_month - 1] else ref_month

    # Return season file with best yield
    df = pd.read_csv("output/%s_M%2.2d/season.txt" % (grid, ref_month),
        sep="\t",
        header=0,
        skiprows=[1],
        skipinitialspace=True,
    )
    df = df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))
    df['crop'] = df['crop'].str.strip()
    df.insert(0, "grid", grid)

    return df


def run_cycles(params):
    # Create input directories
    os.makedirs("input/soil", exist_ok=True)
    os.makedirs("input/weather", exist_ok=True)
    os.makedirs("summary", exist_ok=True)

    first = True

    tmp_max = max_tmps[params["crop"]]
    tmp_min = min_tmps[params["crop"]]
    crop = crops[params["crop"]]

    # Read in look up table
    frames = []
    for id in LUIDS:
        frames.append(pd.read_csv(f"data/{params['crop']}_{id}_v6_Lookup.csv"))
    df = pd.concat(frames)

    # Get a list of grids
    grids = df["nw_cntrl_0"].unique()
    df.set_index(["nw_cntrl_0"], inplace=True)

    # Create operation files
    for month in range(1, 13):
        with open(f"data/template.operation") as op_file:
            op_src = Template(op_file.read())
            op_data = {
                "doy_start": doys[str(month)][0],
                "doy_end": doys[str(month)][1],
                "max_tmp": tmp_max,
                "min_tmp": tmp_min,
                "crop": crops[params["crop"]],
            }
            result = op_src.substitute(op_data)
            with open("./input/M%2.2d.operation" % (month), "w") as f:
                f.write(result + "\n")

    # Run each grid
    for kgrid in range(len(grids)):
        ## Pick one location in the grid for soil file
        grid = grids[kgrid][len("nw_cntrl_03."):-8]

        ## Unzip weather file
        weather = f"{params['scenario']}_{grid}.weather"

        cmd = [
            "unzip",
            "-oj",
            f"data/{params['scenario']}.zip",
            f"{params['scenario']}/{weather}",
            "-d",
            "./input/weather/"
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        ## Some locations in the lookup table are sea points in the CLM grids, which do not have weather files. If so,
        ## skip the grid
        if not os.path.exists(f"input/weather/{weather}"):
            print(f"Skip {grid} due to unavailable weather file")
            continue

        months = calculate_months_for_planting(weather, tmp_max, tmp_min)

        if len(months) == 0:
            print(f"Skip {grid} due to climate")
            continue

        ## Unzip soil file
        n = len(df.loc[grids[kgrid]].shape)
        for kloc in range(n):
            if n == 1:  # Only one location in the grid
                soil = df.loc[grids[kgrid]]["avg_soil_file"]        # This is a bug in the lookup table. Column "avg_soil_file" contains major, and vice versa
            else:
                soil = df.loc[grids[kgrid]]["avg_soil_file"][kloc]

            ### Get land use ID because different land uses are contained in different soil archives
            luid = re.search(f"{params['crop']}_v(.*?)_", soil).group(1)

            cmd = [
                SEVEN_ZIP,
                "e",
                f"data/{params['crop']}{luid}_v9_major.7z",
                "-oinput",
                soil,
                "-aoa",     # Overwrite without prompt
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            ### If soil file exists, exit the loop. Otherwise, test the next location until a soil file is found
            if os.path.exists(f"input/{soil}"):
                break

        ## Test if a soil file has been provided. If not, skip the location
        if not os.path.exists(f"input/{soil}"):
            print(f"Skip {grid} due to unavailable soil file")
            continue
        else:
            print(f"Grid {grid}, soil file: {soil}")

        ## Convert soil files to latest Cycles format
        convert_soil(soil)

        ## Run each month
        for month in months:
            ### Create control file
            with open(f"data/template.ctrl") as ctrl_file:
                ctrl_src = Template(ctrl_file.read())
                ctrl_data = {
                    "start": START_YEAR,
                    "end": END_YEAR,
                    "operation": f"M{month}.operation",
                    #"soil": "soil/%s" % (soil),    # TEMP_DISABLED
                    "soil": "tmp/GenericHagerstown.soil",
                    "weather": f"weather/{weather}",
                }

                result = ctrl_src.substitute(ctrl_data)
                with open(f"./input/{grid}_M{month}.ctrl", "w") as f:
                    f.write(result + "\n")

            ### Run Cycles
            cmd = [
                "./Cycles",
                "-bs",
                f"{grid}_M{month}",
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                print(f"Error running {grid}_M{month}")
                continue

        ## Read season files and find optimal planting months
        exdf = find_optimal_planting_dates(grid, months)

        if exdf.empty:
            print(f"No yield from {grid}.")
            continue

        output_file = f"summary/{params['scenario']}_{params['crop']}.txt"
        if first:
            exdf.to_csv(output_file, index=False)
            first = False
        else:
            exdf.to_csv(output_file, mode="a", header=False, index=False)

        ## Remove generated control files and soil files
        cmd = "rm input/*.soil input/*.ctrl"
        subprocess.run(cmd, shell="True")

        ## Remove generated soil files and weather files
        cmd = "rm input/soil/* input/weather/*"
        subprocess.run(cmd, shell="True")

        ## Remove output files
        cmd = "rm -r output/*"
        subprocess.run(cmd, shell="True")

    # Remove operation files
    cmd = "rm input/*.operation"
    subprocess.run(cmd, shell="True")


def _main():
    parser = argparse.ArgumentParser(description="Cycles execution for a crop")
    parser.add_argument("--crop", default="Maize", help="Crop to be simulated")
    parser.add_argument("--scenario", default="nw_cntrl_03", help="NW scenario")
    args = parser.parse_args()
    run_cycles(vars(args))


if __name__ == "__main__":
    _main()
