import numpy as np

if __name__ == '__main__':
    metric = np.load('E:\\SME\\Data_Science\\AI_challenging_class_peojects'
                     '\\fake_face_detection_220418\\patch-forensics\\results\\pf220421-11\\metrics.npz')
    # print(len(metric['precision']))
    # print(len(metric['recall']))
    # print(len(metric['thresholds']))
    # for i in metric.files:
    #     print(i, ':', metric[i])
    print('acc:', metric['acc'])
    print('ap:', metric['ap'])
    print('n:', metric['n'])

