import subprocess
import time

project_schedule = [
    ('0e856125-d8eb-43db-93c3-27b4c4d0175e', '../../../GNS3/projects/test4/', 'test4', 0),
    ('CaD408D5-8b66-062e-a7Cb-D9aE809b5EcB', '../../../GNS3/projects/test5/', 'test5', 0),
    ('EdcBfad9-64F4-Ecd5-d964-dF8BEB7D23ee', '../../../GNS3/projects/test6/', 'test6', 0),
    ('59D09fAa-451c-67FF-BEDe-d4016F1C650c', '../../../GNS3/projects/test7/', 'test7', 0),
    ('019F4C99-fD13-bbC5-9b50-f10862eeBBEa', '../../../GNS3/projects/test8/', 'test8', 0),
    ('ADee5dB5-BE3d-90Cc-1474-CCe4BD81574A', '../../../GNS3/projects/test9/', 'test9', 1),
    ('854Ee466-7E8A-C0A5-Dfec-A7b6b9fCF45f', '../../../GNS3/projects/test10/', 'test10', 1),
    ('bbcbeb7E-e117-2Dc8-dBCD-EaBC4E51f6A5', '../../../GNS3/projects/test11/', 'test11', 1),
    ('4Cc83205-dEba-9c04-429B-cAd55efcD85c', '../../../GNS3/projects/test12/', 'test12', 1),
    ('8b0Dbe85-f8f7-Eda1-D35C-39836dc0aA1b', '../../../GNS3/projects/test13/', 'test13', 1),

]
options_schedule = [
    ('ospf', 0, 'N'),
    ('ospf', 1, 'N'),
    ('ospf', 2, 'N'),
    ('eigrp', 3, 'Y'),
    ('eigrp', 3, 'N')
]

count = 1
size_table = [['all', '5', '2', ''], ['all', '10', '5', '']]
total_options = len(project_schedule) * len(options_schedule)
for project_id, project_path, project_name, router_count in project_schedule:
    for mode, area_size_desc, auto_sum in options_schedule:
        print(project_id, project_name, mode, area_size_desc, auto_sum)
        print(f'{count}/{total_options}')
        # boot up configurator first.
        proc0 = subprocess.Popen(['python3', './ImagetoGNS3.py', '--gns3_file', 
               f'{project_path}{project_name}.gns3', '--dir', project_path, 
               '--model', '../model_final.pth', '--additional', mode, 
                '--size', size_table[router_count][area_size_desc], 
                '--auto-sum', auto_sum])
        proc0.wait()
        # boot up two processes
        proc1 = subprocess.Popen(['gns3', f'{project_path}{project_name}.gns3'])
        # Wait for a minute for GNS3 gui to boot up
        time.sleep(60)
        proc2 = subprocess.Popen(['python3', 'FastAutoGNS3Test.py', '-n', project_name, 
                                  '-i', project_id, '-f', f'{project_path}project-files', 
                                  '-a', mode, '-o', '../', '-nt', f'{mode}-{size_table[router_count][area_size_desc]}'
                                  ,'--size', size_table[router_count][area_size_desc], 
                                  '--auto-sum', auto_sum], stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        # wait on 2
        proc2.wait()
        # if 2 finishes, after 30 seconds, kill 1
        time.sleep(30)
        proc1.kill()
        # testing schedule
        output2 = proc2.communicate()[0]
        print(output2.decode())
        print()
        count += 1