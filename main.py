## $python main.py c|b 筋力 敏捷 知力 体力 育成回数 ##
import time
import datetime
import sys
import os
import subprocess
import signal
import random
import csv
import tesserocr
import cv2
from PIL import Image, ImageChops
import numpy as np
import re
from string import digits
from paddleocr import PaddleOCR

ss_dir = r"%s\tmp" %(os.getcwd())

# tesserocrの精度がよくないため
ocr2 = PaddleOCR(use_angle_cls=False, lang='en', show_log=False, det=False, use_gpu=False, ir_optim=False, precision='int8', enable_mkldnn=False, det_algorithm='DB', max_text_length=5)

dev_addr = "0.0.0.0"

MAX_OCR_RETRY = 5
NUM_ARGS = 7
NUM_ARGS_DEV = 8
SEC_WAIT_GET_STATUS = 0.5
SEC_RETRY_GET_STATUS_INTERVAL = 1
SEC_RETRY_OCR_INTERVAL = 2.5
SEC_WAIT_SIGINT = 2
IMG_RESIZE_WIDTH = 1.64

CALC_THRESHOLD = 0 #育成計算値チェック
cancel_count = 0
CANCEL_THRESHOLD = 3

res_x=0
res_y=0
##540p point value
preStatusxy = [
    [389, 405,        #y1, y2
    133, 185],        #x1, x2
    [420, 436,
    133, 185],
    [451, 467,
    133, 185],
    [482, 498,
    133, 185]
] 
statusxy = [
    [389, 405,        #y1, y2
    372, 424],        #x1, x2
    [420, 436,
    372, 424],
    [451, 467,
    372, 424],
    [482, 498,
    372, 424]
]
# y1 y2 x1 x2
trainingxy = []

tapxy=[
    [160, 713], #c級/cancel
    [377, 713], #b級/accept
    [160, 810]  #a級
]

TAP_C = 0
TAP_B = 1
TAP_A = 2

firstParam = list()
statusKanji = ["筋力", "敏捷", "知力", "体力"]

def main(args):
    init(args)
    exec_ikusei(args)
    show_result()
    beepExit()

def init(args):
    global dev_addr
    global CALC_THRESHOLD
    now = datetime.datetime.now()

    signal.signal(signal.SIGINT, sigint_handler)
    if len(args) != NUM_ARGS and len(args) != NUM_ARGS_DEV:
        print("err: number of args not matched.")
        print(args)
        exit()
    if len(args) == NUM_ARGS_DEV:
        dev_addr = args[7]
    else:
        dev_str = subprocess.check_output(["nox_adb", "devices"])
        dev_addr = dev_str.decode("utf-8").splitlines()[1].split("\t")[0]
    if not os.path.exists(ss_dir):
        print("err: ss_dir not exist")
        print(ss_dir)
        exit()

    resolution_adjustment()
    img = getStatus()
    for i in range(4):
        statusImg = img.crop(([preStatusxy[i][2], preStatusxy[i][0], preStatusxy[i][3], preStatusxy[i][1]])).convert("RGB")
        firstParam.append(ocr2.ocr(np.asarray(statusImg), cls=False, det=False)[0][0][0])

    calcStatus.preParam = list()
    for i in range(4):
        calcStatus.preParam.append(firstParam[i])
    if args[1] == 'c':
        CALC_THRESHOLD = 15
    elif args[1] == 'b':
        CALC_THRESHOLD = 18
    else:
        CALC_THRESHOLD = 26

def sigint_handler(signal, frame):
    print("---script terminated by SIGINT---")
    time.sleep(SEC_WAIT_SIGINT)
    show_result()
    beepExit()

def exec_ikusei(args):
    global cancel_count
    print("---script start---")
    for i in range(int(args[6])):
        print("%d/%d" %(i+1, int(args[6])))

        tapTraining(args[1])

        calcStatus(args[1], float(args[2]), float(args[3]), float(args[4]), float(args[5]))
        print("-----\n")

        if cancel_count > CANCEL_THRESHOLD:
            print("script stop by cancel_count")
            break

    print("---script end---")

def show_result():
    img = getStatus()
    param_end = list()
    for i in range(4):
        statusImg = img.crop(([preStatusxy[i][2], preStatusxy[i][0], preStatusxy[i][3], preStatusxy[i][1]])).convert("RGB")
        param_end.append(ocr2.ocr(np.asarray(statusImg), cls=False, det=False)[0][0][0])

    print("result:")
    print("筋力：{:+}、敏捷：{:+}、知力：{:+}、体力：{:+}". format(int(param_end[0]) - int(firstParam[0]),
                                                                int(param_end[1]) - int(firstParam[1]),
                                                                int(param_end[2]) - int(firstParam[2]),
                                                                int(param_end[3]) - int(firstParam[3])))

def resolution_adjustment():
    global preStatusxy, statusxy, res_x, res_y
    default_x = 540
    default_y = 960
    res = subprocess.run("nox_adb -s %s shell wm size" %(dev_addr), shell=True, stdout=subprocess.PIPE)
    resol = res.stdout
    if "540" in str(resol):
        res_x = 540
        res_y = 960
        print("warn: 解像度が低すぎるため誤認識率が高くなる可能性があります。(推奨：1080p)")
    elif "720" in str(resol):
        res_x = 720
        res_y = 1280
        print("warn: 解像度が低すぎるため誤認識率が高くなる可能性があります。(推奨：1080p)")
    elif "900" in str(resol):
        res_x = 900
        res_y = 1600
    elif "1080" in str(resol):
        res_x = 1080
        res_y = 1920
    elif "1440" in str(resol):
        res_x = 1440
        res_y = 2560
    elif "2160" in str(resol):
        res_x = 2160
        res_y = 3840
    else :
        print("err: unexpected screen resolution")
        beepExit()

    print(resol)

    ##ajust resolution
    for i in range(4):
        preStatusxy[i][0] = int(preStatusxy[i][0] * res_y / default_y)
        preStatusxy[i][1] = int(preStatusxy[i][1] * res_y / default_y)
        preStatusxy[i][2] = int(preStatusxy[i][2] * res_x / default_x)
        preStatusxy[i][3] = int(preStatusxy[i][3] * res_x / default_x)
        statusxy[i][0] = int(statusxy[i][0] * res_y/default_y)
        statusxy[i][1] = int(statusxy[i][1] * res_y/default_y)
        statusxy[i][2] = int(statusxy[i][2] * res_x/default_x)
        statusxy[i][3] = int(statusxy[i][3] * res_x/default_x)
    for i in range(3):
        tapxy[i][0] = int(tapxy[i][0] * res_x / default_x)
        tapxy[i][1] = int(tapxy[i][1] * res_y / default_y)

def tap(n):
    subprocess.call("nox_adb -s %s shell input touchscreen tap %d %d" %(dev_addr, tapxy[n][0], tapxy[n][1]), shell=True)

def getStatus():
    img = ImageSS_PIL()

    # 画像の2色化
    img_gray = img.convert("L")
    img_bin = img_gray.point(lambda _: 1 if _ > 180 else 0, mode="1")

    return img_bin

def saveStatus():
    img = getStatus()
    img.save(r"%s\fullscreen.png" %(ss_dir), quality=100)
    for i in range(4):
        img_status = img.crop(([preStatusxy[i][2], preStatusxy[i][0], preStatusxy[i][3], preStatusxy[i][1]]))
        img_status.save(r"%s\status_before_%s.png" %(ss_dir, i), quality=100)
        img_status = img.crop(([statusxy[i][2], statusxy[i][0], statusxy[i][3], statusxy[i][1]]))
        img_status.save(r"%s\status_after_%s.png" %(ss_dir, i), quality=100)

    exit()

def tapTraining(type):
    if type == 'c':
        tap(TAP_C)
    elif type == 'b':
        tap(TAP_B)
    elif type == 'a':
        tap(TAP_A)
    else:
        print("err: 1つ目の引数が不正です: %s" %(type))
        beep()
        exit()
    time.sleep(SEC_WAIT_GET_STATUS)

def beepExit():
    print("\007")
    print("\007")
    print("\007")
    exit()

def calcStatus(training, a, b, c, d):
    global CALC_THRESHOLD, cancel_count

    img = getStatus()
    param = list()
    for i in range(4):
        ocr_failure_cnt = 0
        statusImg = img.crop(([statusxy[i][2], statusxy[i][0], statusxy[i][3], statusxy[i][1]]))
        for retryCount in range(MAX_OCR_RETRY):
            ocrValue = ocr2.ocr(np.asarray(statusImg.convert("RGB")), cls=False, det=False)[0][0][0]
            # 育成ボタンが押されていない可能性があるため育成ボタンを押下
            if not ocrValue.isdecimal():
                preStatusImg = img.crop(([preStatusxy[i][2], preStatusxy[i][0], preStatusxy[i][3], preStatusxy[i][1]]))
                preOcrValue = ocr2.ocr(np.asarray(preStatusImg.convert("RGB")), cls=False, det=False)[0][0][0]
                if preOcrValue.isdecimal():
                    tapTraining(training)

                img = getStatus()
                statusImg = img.crop(([statusxy[i][2], statusxy[i][0], statusxy[i][3], statusxy[i][1]]))
                continue

            hasError = False
            try:
                # 育成変動値はCALC_THRESHOLDを超えない
                if abs(int(ocrValue) - int(calcStatus.preParam[i])) > CALC_THRESHOLD or abs(int(ocrValue) - int(calcStatus.preParam[i])) == 0:
                    hasError = True
                    preStatusImg = img.crop(([preStatusxy[i][2], preStatusxy[i][0], preStatusxy[i][3], preStatusxy[i][1]])).convert("RGB")
                    calcStatus.preParam[i] = ocr2.ocr(np.asarray(preStatusImg), cls=False, det=False)[0][0][0]

                    print("%s - %s > %s" %(ocrValue, calcStatus.preParam[i], CALC_THRESHOLD))
            except ValueError:
                hasError = True
                img = getStatus()
                statusImg = img.crop(([statusxy[i][2], statusxy[i][0], statusxy[i][3], statusxy[i][1]]))
                print("%sステータス変動値が数値エラーのためOCR読み込みを再実行します...%d" %(statusKanji[i], ocr_failure_cnt + 1))

            if hasError:
                ocr_failure_cnt += 1
                if ocr_failure_cnt == MAX_OCR_RETRY:
                    print("OCR誤認識検知、%sステータスの読み込みに失敗したため終了します...%d" %(statusKanji[i], ocr_failure_cnt))
                    beepExit()

                continue
            else:
                param.append(ocrValue)

            break

    try:
        calc = (float(param[0]) - float(calcStatus.preParam[0])) * a \
                + (float(param[1]) - float(calcStatus.preParam[1])) * b \
                + (float(param[2]) - float(calcStatus.preParam[2])) * c \
                + (float(param[3]) - float(calcStatus.preParam[3])) * d
    except ValueError:
        print(param)
        beepExit()

    print("筋力(%.2f)：%d\t(%s -> %s)\n敏捷(%.2f)：%d\t(%s -> %s)\n知力(%.2f)：%d\t(%s -> %s)\n体力(%.2f)：%d\t(%s -> %s)" %(
        a, int(param[0]) - int(calcStatus.preParam[0]), calcStatus.preParam[0], param[0],
        b, int(param[1]) - int(calcStatus.preParam[1]), calcStatus.preParam[1], param[1],
        c, int(param[2]) - int(calcStatus.preParam[2]), calcStatus.preParam[2], param[2],
        d, int(param[3]) - int(calcStatus.preParam[3]), calcStatus.preParam[3], param[3]
    ))

    ## 誤認識がなければここまでくる
    print("Calculation Res: %.2f" %calc)

    if calc > 0:
        cancel_count = 0
        print("Accept")
        tap(TAP_B)
        for i in range(4):
            calcStatus.preParam[i] = param[i]
    else:
        ++cancel_count
        print("Cancel")
        tap(TAP_C)

def ImageSS_PIL():
    pipe = subprocess.Popen("nox_adb -s %s exec-out screencap" %(dev_addr), stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    image_bytes = pipe.stdout.read()
    return Image.frombuffer("RGBA", (res_x, res_y), image_bytes, "raw", "RGBA", 0, 1)

if __name__ == '__main__':
    main(sys.argv)
