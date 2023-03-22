import time
from DropLinkTest import main as DLTMain

with open('data_log3.txt', 'w+') as f:
    for test, m, ops in [('test13', 'eigrp', 'Y')]:
        agg_time = []
        for i in range(20):
            execution_mode = m
            special_op = '--size' if m == 'ospf' else '--auto-sum'
            project_file_path = f'../../../GNS3/projects/{test}/project-files'
            output_path = f'../../../GNS3/projects/{test}/'

            print(ops)
            try:
                if m == 'eigrp':
                    time_delta= DLTMain(manual=True, NAME=test, PP_PATH=project_file_path, MODE=m, AUTO_SUM=ops, OUT=output_path)
                else:
                    time_delta= DLTMain(manual=True, NAME=test, PP_PATH=project_file_path, SIZE=ops, MODE=m, OUT=output_path)
                print(f'run {i}', test, m, ops, time_delta)
                f.write(f'run {i}, {test}, {m}, {ops}, {time_delta}\n')
                f.flush()
                agg_time.append(time_delta)
                time.sleep(10)
            except Exception as e:
                print(e)
                continue
        print('average time', test, m, ops, sum(agg_time)/len(agg_time))
        f.write(f'average time, {test}, {m}, {ops}, {sum(agg_time)/len(agg_time)}\n')
        f.flush()
