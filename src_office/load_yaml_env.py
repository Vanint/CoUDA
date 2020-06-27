'''
A simple script to load env variables in yaml
'''
from __future__ import print_function

import os
import argparse

def load_env(fname_yaml):
    '''
    Append env defined in yml file to os.environ
    '''
    fp=open(fname_yaml,'r')
    lines = fp.read().splitlines()
    fp.close()
    flag_env=False
    flag_value=False
    name, value='', ''
    for l in lines:
        if 'env:' in l:
            flag_env=True
            continue
        if flag_env:
            if flag_value:
                if 'value:' in l:
                    value = l.split()[1]
                    os.environ[name] = value
                    print('load env name: %s; value:%s'%(name,value))
                    flag_value = False
                else:
                    break
            else:
                if '- name:' in l:
                    name = l.split()[2]
                    flag_value = True
                else:
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load env defined in yml file.')
    parser.add_argument('file', type=str, help='yml file to load')
    args = parser.parse_args()
    load_env(args.file)
