import sys
import os
import yaml
from datetime import datetime

def generate_id(name):
    init_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    id = f'{name}_{init_time}'

    root_path = sys.path[0]
    dir_path = os.path.join(root_path, 'loggers', 'results', id)
    print(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return id

def find_save_path(id):
    root_path = sys.path[0]
    dir_path = os.path.join(root_path, 'loggers', 'results', id)
    return dir_path

def logger(id, header, contents):
    root_path = sys.path[0]
    file_path = os.path.join(root_path, 'loggers', 'results', id, 'log.yaml')

    contents = {header: contents}

    if os.path.exists(file_path):
        with open(file_path, 'r') as yamlfile:
            current_yaml = yaml.safe_load(yamlfile)
            current_yaml.update(contents)
    else:
        current_yaml = contents
    with open(file_path, 'w') as yamlfile:
        yaml.dump(current_yaml, yamlfile)

def read_log(id, header):
    root_path = sys.path[0]
    file_path = os.path.join(root_path, 'loggers', 'results', id, 'log.yaml')

    with open(file_path, 'r') as yamlfile:
        log = yaml.safe_load(yamlfile)

    return log[header]