"""
Update an existing mcmcfit input file using the results of a
previous chain.
"""

import argparse

import emcee
import h5py
import numpy as np
import pandas as pd

from utils import read_chain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--thin",
        action="store",
        type=int,
        default=5,
        help="thinning for chain (default=5)",
    )
    parser.add_argument(
        "-d",
        "--discard",
        action="store",
        type=int,
        default=0,
        help="thinning for chain (default=5)",
    )
    parser.add_argument("input_file", help="mcmc input file")
    parser.add_argument("chain_file", help="chain output file")
    args = parser.parse_args()

    print("Reading in the chain")
    colkeys, df = read_chain(args.chain_file)
    # get median from chain
    results = df.quantile(0.5, numeric_only=True)

    # strip "core" from core params
    results.index = [entry.replace("_core", "") for entry in results.index]

    # replace lines in MCMC input file
    line_fmt = "{:>10} {:1} {:>16.8f} {:>12} {:>16} {:>16} {:>12}\n"
    mcmc_input = open(args.input_file).readlines()
    for i, line in enumerate(mcmc_input):
        if line.startswith("#"):
            continue
        fields = line.strip().split()  # split line on whitespace
        if len(fields) == 0:
            continue

        parname = fields[0]
        if parname in results.index:
            # we have an updated value for this parameter!
            fields[2] = results[parname]
            mcmc_input[i] = line_fmt.format(*fields)

    with open(args.input_file, "w") as f:
        for line in mcmc_input:
            f.write(line)
