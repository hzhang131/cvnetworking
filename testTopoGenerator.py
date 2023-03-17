import numpy as np
import os
import argparse
import random

SET_NUM = 10

def generate_node_list(name):
    test_dir = './HugeTest/' + name
    if os.path.exists(test_dir) is False:
        os.makedirs(test_dir)
    file_path = test_dir + '/' + name + '.json'
    with open(file_path, 'w') as l:
        l.write('[\n')
        id = 0
        for i in range(SET_NUM):
            l.write('    {"id":' + str(id) + ', "pred_class": 0' + '},\n')
            l.write('    {"id":' + str(id+1) + ', "pred_class": 1' + '},\n')
            if i == SET_NUM - 1:
                l.write('    {"id":' + str(id+2) + ', "pred_class": 4' + '}\n')
            else:
                l.write('    {"id":' + str(id+2) + ', "pred_class": 4' + '},\n')
            id += 3
        l.write(']')

def generate_matrix(name):
    file_path = './HugeTest/' + name + '/'+ name + '.txt'
    connection = {}
    link_count = {}
    for i in range(SET_NUM * 3):
        if i % 3 == 0:
            # a router
            connection[str(i)] = {}
            link_count[str(i)] = 1
    
    for i in range(SET_NUM * 3):
        for j in range(SET_NUM * 3):
            if i % 3 == 0 and j % 3 == 0 and i != j:
                link = connection[str(j)].get(str(i))
                if link is None:
                    if link_count[str(i)] < 4 and link_count[str(j)] < 4:
                        link = random.randint(0, 1)
                        if link == 1:
                            connection[str(j)][str(i)] = 1
                            connection[str(i)][str(j)] = 1
                            link_count[str(i)] += 1
                            link_count[str(j)] += 1
        if i % 3 == 0 and link_count[str(i)] == 1:
            print('not linked, id=', i)
            for j in range(SET_NUM * 3):
                if j % 3 == 0 and i != j:
                    link = connection[str(j)].get(str(i))
                    if link is None:
                        if link_count[str(i)] < 4 and link_count[str(j)] < 4:
                            print("find linkable router")
                            link = 1
                            connection[str(j)][str(i)] = 1
                            connection[str(i)][str(j)] = 1
                            link_count[str(i)] += 1
                            link_count[str(j)] += 1
                            break
        if i % 3 == 0 and link_count[str(i)] == 1:
            print("error, unable to connect router id" + str(i) + ' to topo')
            exit(1)

    with open(file_path, 'w') as m:
        for i in range(SET_NUM * 3):
            line = ''
            for j in range(SET_NUM * 3):
                if i % 3 == 0:
                    if j % 3 == 0 and j != i:
                        link = connection[str(j)].get(str(i))
                        if link is None:
                            link = 0
                        line += str(link)
                    elif j == i + 1:
                        line += '1'
                    else:
                        line += '0'
                elif i % 3 == 1:
                    if j == i - 1 or j == i + 1:
                        line += '1'
                    else:
                        line += '0'
                else:
                    if j == i - 1:
                        line += '1'
                    else:
                        line += '0'
                if j != SET_NUM * 3 - 1:
                    line += ' '
            if i != SET_NUM *3 - 1:
                line += '\n\n'
            m.write(line)
    print(connection)
                         

def parse():
    """
    This function parses the command-line flags

    Parameters: 
      None
    Returns:
      parser.parse_args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', dest="name", type=str,
                        help='testcase name')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    test_name = args.name
    if args.name is None:
        print("please input the name for the test case")
        exit(1)
    generate_node_list(test_name)
    generate_matrix(test_name)