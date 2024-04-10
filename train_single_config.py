import os
import subprocess
import signal
import psutil

#######################3
#    准备训练命令
#######################

ROOT_PATH = "/workspace/auto_paddle_benchmark"
MODEL_PATH = ROOT_PATH + "/PaddleModels"

configs = {
    "nsys_bin" : "nsys",
    "timeline_dir": ROOT_PATH + "/timelines/",
    "json_dir": ROOT_PATH + "/jsons/",
    "analyse_log_dir": ROOT_PATH + "/analyse_json_logs/",
    "repo_root":{
        'PaddleSeg':  MODEL_PATH + "/PaddleSeg/", 
        "PaddleClas": MODEL_PATH + "/PaddleClas/",
        "PaddleDetection": MODEL_PATH + "/PaddleDetection/",
        "PaddleOCR": MODEL_PATH + "/PaddleOCR/",
        "PaddleVideo": MODEL_PATH + "/PaddleVideo/",
        "PaddleGAN": MODEL_PATH + "/PaddleGAN/",
        "PaddleNLP": MODEL_PATH + "/PaddleNLP/tests/",
    },
    "python_train_pattern": ["tools/train.py", "python main.py", "tools/main.py", "test_tipc/train.py", "../examples/language_model/bert/run_pretrain.py"],
}

def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Support Args:")
    parser.add_argument("--name",                 type=str,   default="./data/Amazon_Instant_Video/",  help="data path")
    parser.add_argument("--repo",                 type=str,   default="",  help="data path")
    parser.add_argument("--debug",                action='store_true',  default=False, help="debug mode")
    parser.add_argument("--profile",              action='store_true',  default=False, help="start nvidia profile mode")
    return parser.parse_args()

args = parameter_parser()
if args.repo == "":
    args.repo = args.name.split("_")[0]

print (args)

def get_command_error_file():
    if args.debug: return ROOT_PATH + "/debug.txt"
    else: return "/dev/null"

def clear_debug_file():
    os.system(f"echo '' > {get_command_error_file()}")

root_path = configs['repo_root'][args.repo]
# os.system(f"rm -rf {root_path}test_tipc/data/")
# os.system(f"rm -rf {root_path}dataset/*")
# Get train command and prepare command
child = subprocess.Popen(f"http_proxy='' https_proxy='' python ./get_run_command.py --repo {args.repo} --name {args.name} --type prepare 2>>{get_command_error_file()}", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
child.stdin.close()
info = child.stdout.readlines()
prepare_command = info[0].strip()
child = subprocess.Popen(f"http_proxy='' https_proxy='' python ./get_run_command.py --repo {args.repo} --name {args.name} --type train 2>>{get_command_error_file()}", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
child.stdin.close()
info = child.stdout.readlines()
train_command = info[0].strip()

prepare_command = prepare_command.rstrip(';')
prepare_command = f"cd {root_path} && " + prepare_command + f"1>>{get_command_error_file()} 2>>{get_command_error_file()}"
print ("start prepare dataset: ", prepare_command, flush=True)
exit_code = os.system(prepare_command)
assert exit_code == 0

def set_process_group():
    os.setpgrp()  # 设置子进程的进程组为新的进程组

train_command = f"cd {root_path} && " + train_command 
print ("start get train command: ", train_command, flush=True)
train_proc = subprocess.Popen(train_command, preexec_fn=set_process_group, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
pgid = os.getpgid(train_proc.pid)
import time
process = psutil.Process(train_proc.pid)

def is_child_of(father, child): 
    try:
        for p in child.parents(): 
            if p.pid == father.pid: 
                return True
    except Exception as e:
        pass
    return False

def wait_for_python_process(process):
    def has_train_pattern(string):
        for pattern in configs['python_train_pattern']:
            if pattern in string:
                return True
        return False
    counter = 120
    while counter > 0:
        time.sleep(2)
        counter -= 1
        for proc in psutil.process_iter(['pid', 'name', 'username']):
            if proc.status() == "zombie":
                continue
            if (has_train_pattern(" ".join(proc.cmdline())) and is_child_of(process, proc)):
                return " ".join(proc.cmdline())
    raise RuntimeError("Can not find python process")

def kill_all_child(process):
        # may have exception, we do nothing.
        childs = []
        for proc in psutil.process_iter(['pid', 'name', 'username']):
            if is_child_of(process, proc):
                childs.append(proc)
        for proc in childs:
            try:
                proc.terminate()
            except Exception as e:
                pass

start_command = wait_for_python_process(process) # wait for tools/python appears

# kill may times.
kill_all_child(process)
kill_all_child(process)
kill_all_child(process)

#######################3
#    删除 profile 参数
#######################

def seg_remove_profile_argument(start_command):
    profiler_start = start_command.find("--profiler_options")
    first = start_command.find("'", profiler_start+1)
    second = start_command.find("'", first+1)
    start_command = start_command.replace(start_command[profiler_start:second+1], "")
    return start_command

def clas_remove_profile_argument(start_command):
    profiler_start = start_command.find("--profiler_options")
    return start_command[:profiler_start]

def det_remove_profile_argument(start_command):
    profiler_start = start_command.find("--profiler_options")
    start_point = start_command.find("python")
    return start_command[start_point:profiler_start]

def remove_profile_argument(start_command):
    if args.repo == "PaddleSeg":
        return seg_remove_profile_argument(start_command)
    elif args.repo == "PaddleClas":
        return clas_remove_profile_argument(start_command)
    elif args.repo in ["PaddleDetection", "PaddleOCR", "PaddleVideo", "PaddleGAN", "PaddleNLP"]:
        return clas_remove_profile_argument(start_command)
    else:
        print ("Current start command is : ", start_command)
        raise NotImplementedError("Not support repo: " + args.repo)

start = start_command.find("python")
start_command = start_command[start:]
start_command = remove_profile_argument(start_command)

print ("Base 训练命令:")
print (start_command)


#######################
#    开始训练
#######################

print ("开始训练: ", flush=True)
train_proc.wait()
time.sleep(3)
base_command = start_command

def get_nsys_command(train_command, mode):
    idx = train_command.find("python")
    flags = train_command[0:idx]
    assert idx >= 0 
    nsys_bin = configs['nsys_bin']
    output_report_file = f"--output {configs['timeline_dir']}/{args.name}_{mode}_report"
    nsys_prefix = f"{nsys_bin} profile -t cuda,nvtx -s cpu --cpuctxsw=process-tree --capture-range=cudaProfilerApi --force-overwrite true "
    return f"PROFILE=True {flags} {nsys_prefix} {output_report_file} {train_command[idx:]}", f"{configs['timeline_dir']}/{args.name}_{mode}_report"

def sot_command(base):
    enable_to_static = ast_command(base)
    enable_to_static = enable_to_static.replace("ENABLE_FALL_BACK=False", "ENABLE_FALL_BACK=True")
    return "SOT_LOG_LEVEL=-1 EVENT_LEVEL=-1 COST_MODEL=False " + enable_to_static

def dy_command(base):
    base = base.replace('to_static_training=True', 'to_static_training=False')
    base = base.replace('to_static=True', 'to_static=False')
    base = base.replace('--to_static', '')
    if args.repo in ["PaddleNLP", "PaddleDetection"]: 
        base = base.replace('--to_static', '')
    base = base.replace('Global.to_static=True', 'Global.to_static=False')
    base = base.replace('Global.to_static=true', 'Global.to_static=false')
    base = base.replace('model.to_static=True', 'model.to_static=False')
    return base

def ast_command(base):
    base = base.replace('to_static_training=False', 'to_static_training=True')
    base = base.replace('to_static=False', 'to_static=True')
    if args.repo in ["PaddleNLP", "PaddleDetection"]: 
        base = base.replace('--to_static', '')
        base += " --to_static"
    base = base.replace('Global.to_static=False', 'Global.to_static=True')
    base = base.replace('Global.to_static=false', 'Global.to_static=true')
    base = base.replace('model.to_static=False', 'model.to_static=True')
    return "DY2ST_TEST=True ENABLE_FALL_BACK=False " + base

def train(base, command_fn, mode):
    cmd = command_fn(base)
    if not args.profile: 
        print (f"训练：{mode}")
        cmd = f"cd {root_path} && " + cmd + " 2>&1 " + " | python ~/xkvim/cmd_script/loss_filter.py -n 4 -e 'ips={%f} samp|ips: {%f}' -r mean "+ f" 2>>{get_command_error_file()}"
        print ("Training Command: ", cmd, flush=True)
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        proc.stdin.close()
        try:
            speed = float(proc.stdout.readlines()[0].strip())
        except Exception as e:
            raise RuntimeError(f"Can not get speed, error is {e}")
        return speed
    else: 
        cmd, report_name = get_nsys_command(cmd, mode)
        print (f"训练：{mode}")
        print ("Training Command: ", f"cd {root_path} && " + cmd, flush=True)
        proc = os.system(f"cd {root_path} && " + cmd)

        nsys_bin = configs['nsys_bin']
        json_path = f"{configs['json_dir']}/{args.name}_{mode}_report.json"
        export_json_command = f"{nsys_bin} export {report_name}.nsys-rep --type json --force-overwrite true -o {json_path}"
        proc = os.system(export_json_command)

        ana_log_path = f"{configs['analyse_log_dir']}/{args.name}_{mode}.log"
        analyse_cmd = f"python analyse_json.py {json_path} {ana_log_path}"
        print(f"Start Analyse: {analyse_cmd}")
        proc = os.system(analyse_cmd)

        return "profile done, no speed info."

dy_speed = train(base_command, dy_command, "Dy_Mode")
sot_speed = train(base_command, sot_command, "Sot_Mode")
ast_speed = train(base_command, ast_command, "Ast_Mode")

if not args.profile:
    print ("==============================================")
    print (f"Speed Info for: {args.repo} / {args.name}")
    print ("==============================================")
    print ("SOT    : ", sot_speed)
    print ("AST    : ", ast_speed)
    print ("Dy     : ", dy_speed)

    print ("\n==============================================")
    print (f"Start command for: {args.repo} / {args.name}")
    print ("==============================================")
    print ("SOT    : ", sot_command(base_command))
    print ("AST    : ", ast_command(base_command))
    print ("Dy     : ", dy_command(base_command))
    print("", end="", flush=True)
