import subprocess
import sys

if __name__ == '__main__':
    cnt = 0
    with open('./xxx.txt', 'w+') as f:
        f.write("%d" % cnt)
    while cnt < 500:
        cmd = ['python', 'get_dataset_face_only.py', str(cnt)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        # for line in iter(p.stdout.readline, ''):
        #     print(line),
        #     sys.stdout.flush()
        p.wait()
        with open('./xxx.txt', 'r+') as f:
            cnt = eval(f.read())
        cnt += 1
    # cmd = "python a.py"
    # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # for line in iter(p.stdout.readline, ''):
    #     print(line),
    #     sys.stdout.flush()
