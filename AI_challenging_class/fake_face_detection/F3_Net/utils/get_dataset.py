import cv2
import os
import datetime
import random
import time
import threading
from tqdm import tqdm

LOCAL_VIDEO = 0
CAMERA_STREAM = 1

def getRandomPart():
    """
    获取随机分类
    """
    r = random.random()
    if r < 0.15:
        return 0  # 测试集
    elif 0.15 <= r < 0.35:
        return 1  # 验证集
    else:
        return 2  # 训练集


class dealImgThread(threading.Thread):
    """
    多线程
        为了进行视频处理的时候的图片处理
        处理一张图片，将其截取脸部并且保存
    """
    face_details = []
    details_lock = threading.Lock()
    count_lock = threading.Lock()

    def __init__(self, frame, outPath, timeF, frame_count, image_count):
        threading.Thread.__init__(self)
        self.frame = frame
        self.outPath = outPath
        self.timeF = timeF
        self.frame_count = frame_count
        self.image_count = image_count

    def run(self):
        try:
            dealImgThread.count_lock.acquire()
            cv2.imwrite(self.outPath + "_" + str(self.image_count) + '.png', self.frame)
            dealImgThread.count_lock.release()
        except Exception as e:
            pass


def recognition(vc, outPath, timeF, mode, timeout, total, multiThread=False):
    """

    Args:
        vc: opencv对象
        outPath:
        detector:
        predictor:
        timeF: 视频多少帧进行一次抽帧
        mode:
        timeout:
        total:
        multiThread:

    Returns:

    """
    # 加载模型
    begin_time = time.time()
    if vc.isOpened():
        res = True
    else:
        res = False
    image_count = 0  # 图片计数
    frame_count = 0  # 帧数计数
    while res:
        if mode:
            now_time = time.time()
            if now_time - begin_time > timeout:
                break
        res, frame = vc.read()  # 分帧读取视频

        if not res:
            break
        # cv2.imshow("img", frame)
        frame_count += 1
        if frame_count % timeF == 0:
            if multiThread:
                dealImgThread(frame, outPath, timeF, frame_count, image_count).start()
                image_count += 1
            else:
                try:
                    dealImgThread.count_lock.acquire()
                    cv2.imwrite(outPath + "_" + str(image_count) + '.png', frame)
                    dealImgThread.count_lock.release()
                except Exception as e:
                    pass

        if mode and image_count >= total:
            break
        if multiThread and mode and dealImgThread.image_count >= total:
            break
        # if cv2.waitKey(33) == 27:
        #     break
    vc.release()


def dealLocalVideos(inPath, outPath_list, timeF, multiThread=True, total=10000):
    """

    """
    # 预处理
    for outPath in outPath_list:
        if not os.path.exists(outPath):
            os.makedirs(outPath)

    start = 0
    videos = sorted(os.listdir(inPath))
    for video in tqdm(videos[start:]):
        # print(f"第{start}组开始。。。")
        video = os.path.join(inPath, video)
        print(video)
        img_dir = outPath_list[getRandomPart()] + video.split('/')[-1].split('.')[0]
        print(img_dir)
        vc = cv2.VideoCapture(video)
        recognition(vc, img_dir, timeF, LOCAL_VIDEO, timeout=float("inf"),
                    total=total, multiThread=multiThread)
        start += 1


if __name__ == '__main__':
    # 处理后图片存放位置
    out_dir_list = ['/home/PointCloud/F3_Net2/celeb_raw/test/real/',
                    '/home/PointCloud/F3_Net2/celeb_raw/valid/real/',
                    '/home/PointCloud/F3_Net2/celeb_raw/train/real/']
    print(out_dir_list[getRandomPart()])

    # in_dir = r'/home/PointCloud/F3_Net2/dataset_ffpp/original_sequences/actors/c40/videos'
    # dealLocalVideos(in_dir, out_dir_list, timeF=30, multiThread=True, total=12000)

    in_dir = r'/home/PointCloud/F3_Net2/celeb_raw/Celeb-real/'
    dealLocalVideos(in_dir, out_dir_list, timeF=30, multiThread=True, total=12000)

    in_dir = r'/home/PointCloud/F3_Net2/celeb_raw/celeb-syn/'
    out_dir_list = ['/home/PointCloud/F3_Net2/celeb_raw/test/fake/',
                    '/home/PointCloud/F3_Net2/celeb_raw/valid/fake/',
                    '/home/PointCloud/F3_Net2/celeb_raw/train/fake/']
    dealLocalVideos(in_dir, out_dir_list, timeF=30, multiThread=True, total=5000)

    # in_dir = r'/home/PointCloud/F3_Net2/dataset_ffpp/manipulated_sequences/Face2Face/c40/videos/'
    # out_dir_list = ['/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_raw/test/fake/Face2Face/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_raw/valid/fake/Face2Face/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_raw/train/fake/Face2Face/']
    # dealLocalVideos(in_dir, out_dir_list, timeF=30, multiThread=True, total=5000)
    #
    # in_dir = r'/home/PointCloud/F3_Net2/dataset_ffpp/manipulated_sequences/FaceSwap/c40/videos/'
    # out_dir_list = ['/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_raw/test/fake/FaceSwap/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_raw/valid/fake/FaceSwap/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_raw/train/fake/FaceSwap/']
    # dealLocalVideos(in_dir, out_dir_list, timeF=30, multiThread=True, total=5000)
    #
    # in_dir = r'/home/PointCloud/F3_Net2/dataset_ffpp/manipulated_sequences/NeuralTextures/c40/videos/'
    # out_dir_list = ['/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_raw/test/fake/NeuralTextures/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_raw/valid/fake/NeuralTextures/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_raw/train/fake/NeuralTextures/']
    # dealLocalVideos(in_dir, out_dir_list, timeF=30, multiThread=True, total=5000)
