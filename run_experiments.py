#!/usr/bin/env python3

import argparse
import os
import shlex
import subprocess
import sys
import threading

import torch


class GPU:
    def __init__(self, id, manager):
        self.id = id
        self.manager = manager

    def release(self):
        self.manager.release_gpu(self)

    def __str__(self):
        return str(self.id)


class GPUManager:
    """Track available GPUs and provide on request"""

    def __init__(self, _available_gpus, experiments_per_gpu=1):
        self.semaphore = threading.BoundedSemaphore(len(_available_gpus) * experiments_per_gpu)
        self.gpu_dict_lock = threading.Lock()
        self.available_gpus = dict()
        for available_gpu in _available_gpus:
            self.available_gpus[available_gpu] = experiments_per_gpu

    def get_gpu(self):
        self.semaphore.acquire()
        with self.gpu_dict_lock:
            gpu = None
            for gpu_id, gpu_amount in self.available_gpus.items():
                if gpu_amount > 0:
                    gpu = gpu_id
                    break
            self.available_gpus[gpu] -= 1
        return GPU(gpu, self)

    def get_gpus(self, num_gpu=1):
        gpu_list = []
        for ii in range(num_gpu):
            gpu_list.append(self.get_gpu())
        return gpu_list

    def release_gpu(self, gpu):
        with self.gpu_dict_lock:
            self.available_gpus[gpu.id] += 1
            self.semaphore.release()


def run_command_with_gpus(command, gpu_list):
    print(f'GPU {",".join([str(gpu) for gpu in gpu_list])}: {command}')

    def run_and_release(command, gpu_list):
        myenv = os.environ.copy()
        myenv["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in gpu_list])
        proc = subprocess.Popen(args=command, shell=True, env=myenv)
        proc.wait()
        for gpu in gpu_list:
            gpu.release()

    thread = threading.Thread(target=run_and_release, args=(command, gpu_list))
    thread.start()
    return thread


def run_command_list(manager, command_list, num_gpu):
    for command in command_list:
        gpu_list = manager.get_gpus(num_gpu=num_gpu)
        run_command_with_gpus(command, gpu_list)


def read_commands(exp_file):
    with open(exp_file, "r") as f:
        command_list = [line.rstrip() for line in f]

        # Remove comments and empty lines
        command_list = [x for x in command_list if len(x) > 0]
        command_list = [x for x in command_list if x[0] != "#"]

    return command_list


def expand_repeats(command_list, start_number):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--test_split_index_range", type=int, default=1)
    parser.add_argument("--exp_name", type=str, required=True)

    out_commands = []
    for command in command_list:
        args, _ = parser.parse_known_args(shlex.split(command))
        for repeat_it in range(args.num_repeats):
            new_command = command
            exp_num = repeat_it + start_number
            seed_num = int(exp_num // args.test_split_index_range)
            index_num = int(exp_num % args.test_split_index_range)

            new_command += " --group_name " + args.exp_name
            new_command += " --exp_name " + args.exp_name + "-" + str(index_num) + "-" + str(seed_num)

            new_command += " --test_split_index " + str(index_num)
            new_command += " --random_seed " + str(seed_num)
            out_commands.append(new_command)
    return out_commands


def main():
    parser = argparse.ArgumentParser(description="Schedule a list of GPU experiments.")
    parser.add_argument(
        "-e", "--exp_txt", type=str, required=True, help="txt file with one line per command, see e.g. exp/example.txt"
    )
    parser.add_argument(
        "-g",
        "--gpus",
        nargs="+",
        type=str,
        default=[],
        required=False,
        help="which GPUs to use. If unset, will use all",
    )
    parser.add_argument(
        "-s",
        "--start_number",
        type=int,
        default=0,
        help="starting number for random seed and experiment name for multi-run experiments (default 0)",
    )
    parser.add_argument(
        "--experiments_per_gpu", type=int, default=1, help="number of experiments to run on each GPU (default 1)"
    )
    parser.add_argument("-n", "--num_gpu", type=int, default=1, help="number of GPUs to use per experiment (default 1)")
    args = parser.parse_args()

    gpus = args.gpus
    if len(gpus) == 0:
        # find all available gpus
        gpus = [str(x) for x in range(torch.cuda.device_count())]

    manager = GPUManager(gpus, args.experiments_per_gpu)
    exp_file = args.exp_txt
    command_list = read_commands(exp_file)
    command_list = expand_repeats(command_list, args.start_number)
    run_command_list(manager, command_list, args.num_gpu)


if __name__ == "__main__":
    main()
