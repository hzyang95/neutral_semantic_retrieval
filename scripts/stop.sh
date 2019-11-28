ps aux | grep serve_starter  | grep -v grep | awk '{print $2}'  | xargs kill -9
