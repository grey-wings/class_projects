import cv2
import dlib
import random
import os
import datetime
import time
import threading
import argparse

from tqdm import tqdm

dlib_classifier_path = "./shape_predictor_68_face_landmarks.dat"  # 人脸识别模型路径
LOCAL_VIDEO = 0
CAMERA_STREAM = 1


def getArgs():
    args = argparse.ArgumentParser()
    args.add_argument('start', type=int)
    return args.parse_args()


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

    def __init__(self, frame, outPath, detector, predictor, timeF, frame_count, image_count):
        threading.Thread.__init__(self)
        self.frame = frame
        self.outPath = outPath
        self.detector = detector
        self.predictor = predictor
        self.timeF = timeF
        self.frame_count = frame_count
        self.image_count = image_count

    def run(self):
        dots = self.detector(self.frame, 1)
        backup = dealImgThread.face_details[:]

        for k, d in enumerate(dots):
            shape = self.predictor(self.frame, d)
            # 排除静态人脸,如画像
            isSame = False
            for i in range(len(backup)):
                same_count = 0
                for p_pt, n_pt in zip(backup[i].parts(), shape.parts()):
                    if p_pt.x == n_pt.x and p_pt.y == n_pt.y:
                        same_count += 1
                if same_count >= 10:
                    isSame = True
                    break

            if self.frame_count == self.timeF:
                dealImgThread.details_lock.acquire()
                dealImgThread.face_details.append(shape)
                dealImgThread.details_lock.release()

            if not isSame and self.frame_count != self.timeF:
                height, width = self.frame.shape[:2]

                multiple = max(d.height(), d.width()) / 1.9
                need_adjust_height = 0
                need_adjust_width = 0
                difference = abs(d.height() - d.width())
                if d.height() > d.width():
                    need_adjust_width = 1
                else:
                    need_adjust_height = 1

                height_overflow = False
                width_overflow = False
                if height < width and d.height() + multiple * 2 + need_adjust_height * difference > height:
                    height_overflow = True

                if width < height and d.width() + multiple * 2 + need_adjust_width * difference > width:
                    width_overflow = True

                if width_overflow:
                    backup = multiple
                    multiple = (height - d.width()) / 2
                    need_adjust_height = 0

                top_cross = max(0 - d.top() + multiple + need_adjust_height * difference / 2, 0)
                bottom_cross = max(d.bottom() + multiple + need_adjust_height * difference / 2 - height, 0)
                top = max(d.top() - multiple - need_adjust_height * difference / 2 - bottom_cross, 0)
                bottom = min(d.bottom() + multiple + need_adjust_height * difference / 2 + top_cross, height)

                if width_overflow:
                    multiple = backup

                if height_overflow:
                    multiple = (height - d.width()) / 2
                    need_adjust_width = 0

                left_cross = max(0 - d.left() + multiple + need_adjust_width * difference / 2, 0)
                right_cross = max(d.right() + multiple + need_adjust_width * difference / 2 - width, 0)
                left = max(d.left() - multiple - need_adjust_width * difference / 2 - right_cross, 0)
                right = min(d.right() + multiple + need_adjust_width * difference / 2 + left_cross, width)

                save_img = self.frame[int(top):int(bottom), int(left):int(right)]

                try:
                    save_img = cv2.resize(save_img, (380, 380))
                    dealImgThread.count_lock.acquire()
                    cv2.imwrite(self.outPath + "_" + str(self.image_count) + '.png', save_img)
                    dealImgThread.count_lock.release()
                except Exception as e:
                    print(e)


def setModelPath(path):
    global dlib_classifier_path
    dlib_classifier_path = path


def loadModel():
    global dlib_classifier_path
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_classifier_path)
    return detector, predictor


def recognition(vc, outPath, detector, predictor, timeF, mode, timeout=60, total=100, multiThread=False):
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
    if mode:
        begin_time = time.time()
    if not os.path.exists(outPath):  # 如果不存在就创建文件夹
        os.mkdir(outPath)
    if vc.isOpened():
        res = True
    else:
        res = False
    image_count = 0  # 图片计数
    frame_count = 0  # 帧数计数
    face_details = []  # 辅助排除静态人脸
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
                dealImgThread(frame, outPath, detector, predictor, timeF, frame_count, image_count).start()
                image_count += 1
            else:
                dots = detector(frame,  1)
                backup = face_details[:]
                face_details.clear()
                for k, d in enumerate(dots):
                    shape = predictor(frame, d)

                    # 排除静态人脸,如画像
                    isSame = False
                    for i in range(len(backup)):
                        same_count = 0
                        for p_pt, n_pt in zip(backup[i].parts(), shape.parts()):
                            if p_pt.x == n_pt.x and p_pt.y == n_pt.y:
                                same_count += 1
                        if same_count >= 10:
                            isSame = True
                            break
                    face_details.append(shape)

                    if not isSame and frame_count != timeF:

                        height, width = frame.shape[:2]

                        multiple = max(d.height(), d.width()) / 1.9
                        need_adjust_height = 0
                        need_adjust_width = 0
                        difference = abs(d.height() - d.width())
                        if d.height() > d.width():
                            need_adjust_width = 1
                        else:
                            need_adjust_height = 1

                        height_overflow = False
                        width_overflow = False
                        if height < width and d.height() + multiple * 2 + need_adjust_height * difference > height:
                            height_overflow = True

                        if width < height and d.width() + multiple * 2 + need_adjust_width * difference > width:
                            width_overflow = True

                        if width_overflow:
                            backup = multiple
                            multiple = (height - d.width()) / 2
                            need_adjust_height = 0
                        top_cross = max(0 - d.top() + multiple + need_adjust_height * difference / 2, 0)
                        bottom_cross = max(d.bottom() + multiple + need_adjust_height * difference / 2 - height, 0)
                        top = max(d.top() - multiple - need_adjust_height * difference / 2 - bottom_cross, 0)
                        bottom = min(d.bottom() + multiple + need_adjust_height * difference / 2 + top_cross, height)

                        if width_overflow:
                            multiple = backup
                        if height_overflow:
                            multiple = (height - d.width()) / 2
                            need_adjust_width = 0
                        left_cross = max(0 - d.left() + multiple + need_adjust_width * difference / 2, 0)
                        right_cross = max(d.right() + multiple + need_adjust_width * difference / 2 - width, 0)
                        left = max(d.left() - multiple - need_adjust_width * difference / 2 - right_cross, 0)
                        right = min(d.right() + multiple + need_adjust_width * difference / 2 + left_cross, width)

                        save_img = frame[int(top):int(bottom), int(left):int(right)]
                        # cv2.imshow("save", save_img)
                        try:
                            image_count += 1
                            save_img = cv2.resize(save_img, (380, 380))
                            cv2.imwrite(outPath + "_" + str(image_count) + '.png', save_img)
                        except Exception as e:
                            print(e)
        if mode and image_count >= total:
            break
        if multiThread and mode and dealImgThread.image_count >= total:
            break
        # if cv2.waitKey(33) == 27:
        #     break
    vc.release()


def dealLocalVideos(inPath, outPath_list, timeF=30, multiThread=False, total=10000, start=0):
    # 预处理
    for outPath in outPath_list:
        if not os.path.exists(outPath):
            os.mkdir(outPath)

    # 加载模型
    detector, predictor = loadModel()

    # 视频处理
    videos = sorted(os.listdir(inPath))
    for video in tqdm(videos[start:]):
        # print(f"第{start}组开始。。。")
        video = os.path.join(inPath, video)
        print(video)
        img_dir = outPath_list[getRandomPart()] + video.split('/')[-1].split('.')[0]
        print(img_dir)
        vc = cv2.VideoCapture(video)
        try:
            recognition(vc, img_dir, detector, predictor, timeF, LOCAL_VIDEO, total=total, multiThread=multiThread)
        except:
            print("ERROR")
            continue

        # with open('./xxx.txt', 'r+') as f:
        #     x = eval(f.read())
        # x += 1
        # with open('./xxx.txt', 'w+') as f:
        #     f.write("%s" % x)
        # start += 1


if __name__ == '__main__':
    # args = getArgs()
    out_dir_list = ['/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/test/real/',
                    '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/valid/real/',
                    '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/train/real/']
    print(out_dir_list[getRandomPart()])

    in_dir = r'/home/PointCloud/F3_Net2/dataset_ffpp/original_sequences/actors/c40/videos'
    dealLocalVideos(in_dir, out_dir_list, timeF=30, multiThread=True, total=12000)


    # in_dir = r'/home/PointCloud/F3_Net2/dataset_ffpp/original_sequences/youtube/c40/videos'
    # dealLocalVideos(in_dir, out_dir_list, timeF=30, multiThread=True, total=8000)
    #
    # in_dir = r'/home/PointCloud/F3_Net2/dataset_ffpp/manipulated_sequences/Deepfakes/c40/videos/'
    # out_dir_list = ['/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/test/fake/Deepfakes/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/valid/fake/Deepfakes/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/train/fake/Deepfakes/']
    # dealLocalVideos(in_dir, out_dir_list, timeF=30, multiThread=True, total=5000)
    #
    # in_dir = r'/home/PointCloud/F3_Net2/dataset_ffpp/manipulated_sequences/Face2Face/c40/videos/'
    # out_dir_list = ['/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/test/fake/Face2Face/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/valid/fake/Face2Face/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/train/fake/Face2Face/']
    # dealLocalVideos(in_dir, out_dir_list, timeF=30, multiThread=True, total=5000)
    #
    # in_dir = r'/home/PointCloud/F3_Net2/dataset_ffpp/manipulated_sequences/FaceSwap/c40/videos/'
    # out_dir_list = ['/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/test/fake/FaceSwap/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/valid/fake/FaceSwap/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/train/fake/FaceSwap/']
    # dealLocalVideos(in_dir, out_dir_list, timeF=30, multiThread=True, total=5000)
    #
    # in_dir = r'/home/PointCloud/F3_Net2/dataset_ffpp/manipulated_sequences/NeuralTextures/c40/videos/'
    # out_dir_list = ['/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/test/fake/NeuralTextures/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/valid/fake/NeuralTextures/',
    #                 '/home/PointCloud/F3_Net2/dataset_ffpp/ffpp_face_only/train/fake/NeuralTextures/']
    # dealLocalVideos(in_dir, out_dir_list, timeF=30, multiThread=True, total=5000)


