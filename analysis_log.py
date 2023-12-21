import pandas
import os
import sys

def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Support Args:")
    parser.add_argument("--input",                 type=str,   default="output.log",  help="a input file")
    parser.add_argument("--output",                type=str,   default="output.png",  help="a input file")
    return parser.parse_args()

args = parameter_parser()

with open(args.input, "r") as fp :
    lines = fp.readlines()

statistics = {}

for idx, line in enumerate(lines):
    if 'Speed Info for' in line:
        repo, name = line.split('Speed Info for: ')[1].split('/')
        item = {}
        item['sot'] = float(lines[idx+2].split(':')[1].strip())
        item['ast'] = float(lines[idx+3].split(":")[1].strip())
        item['dy'] =  float(lines[idx+4].split(":")[1].strip())
        statistics[(repo, name)] = item

data_frame = {'repo': [], 'name': [], 'sot': [], 'ast': [], 'dy': [], 'rel_sot_dy': []}
for (repo, name), speed in statistics.items():
    data_frame['repo'].append(repo)
    data_frame['name'].append(name)
    data_frame['sot'] .append(speed['sot'])
    data_frame['ast'] .append(speed['ast'])
    data_frame['dy'] .append(speed['dy'])
    data_frame['rel_sot_dy'] .append(speed['sot'] / speed['dy'] - 1)

data_frame = pandas.DataFrame(data_frame)
if args.output.split(".")[1] == 'xlsx': 
    data_frame.to_excel(args.output, index=True)
elif args.output.split(".")[1] == 'png': 
    sot_speed = data_frame['sot']
    dy_speed = data_frame['dy']
    relative = ((sot_speed - dy_speed) / dy_speed * 100).to_numpy()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors
    from matplotlib.ticker import PercentFormatter
    counts, bins = np.histogram(relative, 20)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.stairs(counts, bins)
    plt.savefig(args.output, transparent=False, dpi=80, bbox_inches="tight")
else:
    raise NotImplementedError("Not a support output format, only support 'png' and 'xlsx'")