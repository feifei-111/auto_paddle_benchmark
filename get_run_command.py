import os
def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Support Args:")
    parser.add_argument("--name",                 type=str,   default=None,  help="data path")
    parser.add_argument("--repo",                 type=str,   default="PaddleClas",  help="data path")
    parser.add_argument("--type",                 type=str,   default="train",  help="data path")
    return parser.parse_args()

args = parameter_parser()
assert args.repo in ['PaddleClas', 'PaddleSeg', 'PaddleDetection', "PaddleOCR", "PaddleVideo", "PaddleGAN", "PaddleNLP"]
assert args.type in ['train', 'prepare']

with open("./run_model_sh.log", "r") as fp :
    lines = fp.readlines()

name = args.name
founded = False
for line in lines: 
    if name in line and args.repo in line and "N1C1" in line: 
        founded = True
        break

if founded is not True: 
    print ("Not Found")
else:
    line = line.strip()
    #print (f"wget {line} -O ./logs/tmp")
    os.system(f"http_proxy='' https_proxy='' wget -O ./logs/tmp {line} ")
    with open("./logs/tmp", "r") as fp :
        lines = fp.readlines()
    if args.type == 'train': 
        print (lines[-1].strip())
    else:
        print (lines[-2].strip())
        