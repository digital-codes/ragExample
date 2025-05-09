import subprocess
import time
import argparse
import os
import sys
import atexit
import signal


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUPERVISOR_CONF = os.path.join(BASE_DIR, "..", "sv", 'supervisord.conf')
SUPERVISORCTL = 'supervisorctl'
SUPERVISORD = 'supervisord'

def wait_for_services():
    while True:
        try:
            output = subprocess.check_output(
                [SUPERVISORCTL, '-c', SUPERVISOR_CONF, 'status'],
                stderr=subprocess.STDOUT
            ).decode()
            lines = output.strip().splitlines()
            not_ready = [line for line in lines if 'RUNNING' not in line]
            if not not_ready:
                print("All services are RUNNING.")
                break
            print("Waiting for services:")
            print("\n".join(not_ready))
            time.sleep(2)
        except subprocess.CalledProcessError as e:
            print("Waiting for supervisor socket to become ready...")
            time.sleep(1)

def start_supervisord():
    print("Starting supervisord...")
    subprocess.Popen([SUPERVISORD, '-c', SUPERVISOR_CONF])
    atexit.register(shutdown_supervisord)

def shutdown_supervisord():
    print("Shutting down supervisord...")
    try:
        subprocess.run([SUPERVISORCTL, '-c', SUPERVISOR_CONF, 'shutdown'],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("Error shutting down supervisor:", e)

def sigint_handler(signum, frame):
    print("\\nCaught Ctrl-C (SIGINT)")
    shutdown_supervisord()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', required=True)
    parser.add_argument('-c', required=True)
    parser.add_argument('-P', required=True)
    parser.add_argument('-p', required=True)
    parser.add_argument('-s', required=True)
    parser.add_argument('-b', action='store_true')

    args = parser.parse_args()

    signal.signal(signal.SIGINT, sigint_handler)

    start_supervisord()
    wait_for_services()

    cmd = [
        sys.executable, os.path.join(BASE_DIR, "..","rag",'ragRemoteQuery.py'),
        '-d', args.d,
        '-c', args.c,
        '-P', args.P,
        '-p', args.p,
        '-s', args.s,
    ]
    if args.b:
        cmd.append('-b')

    print("Running RAG query:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
