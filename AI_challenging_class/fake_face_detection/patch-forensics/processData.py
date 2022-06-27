import shutil
import os

path = '/home/PointCloud/patch-forensics/datasets/real_and_fake_face_testing/real_and_fake_face_testing/'
real_path = '/home/PointCloud/patch-forensics/datasets/real_and_fake_face_testing/real/'
fake_path = '/home/PointCloud/patch-forensics/datasets/real_and_fake_face_testing/fake/'
with open('/home/PointCloud/patch-forensics/label.txt', 'r') as f:
        labels = f.readlines()
        filelist = os.listdir(path)
        labels = [eval(i.strip()) for i in labels]
        for i in filelist:
            idx = eval(i[3:-4]) - 1
            if labels[idx] == 1:
                shutil.copy(path + '/' + i, real_path + '/' + i)
            elif labels[idx] == -1:
                shutil.copy(path + '/' + i, fake_path + '/' + i)
