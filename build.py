import os
import subprocess
import shutil

def build(source, dest, config, env_list, binary):
    dest_real_path = os.path.realpath(dest)
    config_real_path = os.path.realpath(config)

    env_repro = " ".join(env_list)

    shutil.copy(config_real_path, os.path.join(dest_real_path, ".config"))

    threads = int(subprocess.check_output("nproc"))

    data_time_output = os.path.join(dest_real_path, "__time")
    cmdl = [env_repro, f"/usr/bin/time -pq -o {data_time_output}", "make"]
    if source != dest:
        cmdl.append(f"O={dest_real_path}")
    cmdl.append(f"-j{threads}")
    cmds = " ".join(cmdl)

    with open(os.path.join(dest_real_path, "__build_cmd"), 'w') as stream:
        stream.write(cmds)

    return_content = subprocess.run(cmds,
                                    capture_output=True,
                                    shell=True,
                                    cwd=source)

    data_exit_status_output = os.path.join(dest_real_path, "__exit_status")
    data_stdout_output = os.path.join(dest_real_path, "__stdout")
    data_stderr_output = os.path.join(dest_real_path, "__stderr")
    with open(data_exit_status_output, 'w') as output:
        output.write(str(return_content.returncode))
    with open(data_stdout_output, 'wb') as output:
        output.write(return_content.stdout)
    with open(data_stderr_output, 'wb') as output:
        output.write(return_content.stderr)

    duration = None
    with open(data_time_output, 'r') as stream:
        duration = float(stream.readlines()[0].strip().split()[-1])

    return os.path.isfile(os.path.join(dest_real_path, binary)), duration


def randconfig(source, preset=None):
    cmdl = []
    if preset:
        preset_real_path = os.path.realpath(preset)
        cmdl.append(f"KCONFIG_CONFIG={preset_real_path}")
    cmdl.append("make randconfig")
    cmd = " ".join(cmdl)
    return_content = subprocess.run(cmd,
                                capture_output=True,
                                shell=True,
                                cwd=source)


def distclean(source):
    return_content = subprocess.run("make distclean",
                                capture_output=True,
                                shell=True,
                                cwd=source)
