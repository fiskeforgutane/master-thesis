import subprocess
import threading
import time
from tqdm import tqdm

def mowi_instances():
    result = subprocess.run('find ~/mowilib/real/t*/*/*.json', capture_output=True, shell=True)
    assert result.returncode == 0
    return [path for path in result.stdout.decode().splitlines()]

def mirplib_instances(max_t = 180):
    result = subprocess.run('find ~/mirplib/t*/*.json', capture_output=True, shell=True)
    assert result.returncode == 0
    return [path for path in result.stdout.decode().splitlines() if planning_horizon(path) <= max_t]

def planning_horizon(path):
    return next(int(x[1:]) for x in path.split('/') if x[:1] == 't' and x[1:].isdigit())

def remote_solve(host, path):
    timesteps = planning_horizon(path)
    smolt = f'~/master-thesis/target/release/master --problem {path} --log info --termination "1800 no-improvement no-violation & 10800 timeout |" rolling-horizon --full-penalty-after 360000 --population 3 --children 3 --tournament 2 --step-length 2 --subproblem-size {timesteps} --mutation "lite" --travel-at-cap 0 --travel-empty 0 2>> log-$(hostname).txt'
    command = f"ssh -t {host} 'module load gurobi Python && cd /storage/users/akselbor/ && {smolt}'"
    result = subprocess.run(command, shell=True, capture_output=True)
    assert result.returncode == 0

def solve(hosts, paths, f = lambda host, path: time.sleep(1)):
    lock = threading.Lock()
    progress = tqdm(total=len(paths))

    def worker(host):
        def inner():
            while True:
                with lock:
                    if paths:
                        path = paths.pop()
                    else:
                        return

                f(host, path)

                with lock:
                    progress.update(1)
        
        return inner

    threads = [threading.Thread(name=host, target=worker(host)) for host in hosts]

    # We add a delay between each start to avoid a sudden power spike (don't think it should matter, but yeah)
    for thread in threads:
        thread.start()
        time.sleep(2.0)

    # Wait for all problems to be solved before exiting
    for thread in threads:
        thread.join()

    progress.close()
