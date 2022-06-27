import os

if __name__ == '__main__':
    path = r'E:\SME\Data_Science\AI_challenging_class_peojects\F3_Net2\datasets\our_test_2\fake'
    l = os.listdir(path)
    for i in range(len(l)):
        os.rename(os.path.join(path, l[i]), os.path.join(path, 'fake_' + str(i) + '.jpg'))
