import os, errno

#### Files and Directories ####

def create_directory(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

#### Logging Dictionary Tools ####

def add_record(key, value, global_logs):
    if 'logs' not in global_logs['info']:
        global_logs['info']['logs'] = {}
    logs = global_logs['info']['logs']
    split_path = key.split('.')
    current = logs
    for p in split_path[:-1]:
        if p not in current:
            current[p] = {}
        current = current[p]

    final_key = split_path[-1]
    if final_key not in current:
        current[final_key] = []
    entries = current[final_key]
    entries.append(value)

def log_record_dict(usage, log_dict, global_logs):
    for log_key, value in log_dict.items():
        add_record('{}.{}'.format(usage, log_key), value, global_logs)

    