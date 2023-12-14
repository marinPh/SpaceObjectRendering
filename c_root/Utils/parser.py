import sys


def pars_2_args(sys_argv):
    if len(sys_argv) < 3:
        print("Usage: python script.py arg1 arg2")
    else:
        arg1 = sys_argv[-2]
        arg2 = sys_argv[-1]
        print(f"Argument 1: {arg1}")
        print(f"Argument 2: {arg2}")
    return arg1, arg2


def pars_1_arg(sys_argv):
    if len(sys_argv) < 2:
        print("Usage: python script.py arg1")
    else:
        arg1 = sys_argv[-1]
        print(f"Argument 1: {arg1}")
    return arg1
