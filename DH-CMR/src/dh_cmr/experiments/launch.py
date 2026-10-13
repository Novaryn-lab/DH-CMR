import shlex
import subprocess
import sys


def launch(config_path, overrides, dry_run=False):
    command = [sys.executable, "-m", "dh_cmr.cli.train", "--config", str(config_path)]
    for setting in overrides:
        command.extend(["--set", setting])
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)
