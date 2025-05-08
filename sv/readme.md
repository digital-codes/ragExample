
start_main starts and stops compagions 
use: sv/run/start_main.sh

monitor:
supervisorctl -c supervisord.conf status 


All companion processes are started first via autostart=true.

Main program is started last (due to higher priority=30).

stopasgroup=true ensures when main_rag exits, all its subprocesses are cleaned up.

Use supervisorctl shutdown or ctrl+c (if foreground) to tear everything down.

