import cv2
import mediapipe as mp
import time
import math
import numpy as np
import time
ref_img = None  # 참고이미지


def Trackbar():
    global ref_img

    def onChange(x):
        global ref_img
        # print('position changed')
        if x == 1:
            ref_img = cv2.imread("TREE-W.png")
        else:
            ref_img = cv2.imread("TREE-M.png")

    # img = cv2.imread("TREE-M.png")

    cv2.namedWindow("Pose Estimation")
    cv2.createTrackbar('Man or Woman', "Pose Estimation", 0, 1, onChange)
    # while True:

    WM = cv2.getTrackbarPos('Man or Woman', "Pose Estimation")
    if WM == 1:
        ref_img = cv2.imread("TREE-W.png")
    else:
        ref_img = cv2.imread("TREE-M.png")
    cv2.imshow("Pose Estimation", ref_img)


class poseDetector():
    def __init__(self, mode=False, complexity=1, landmarks=True, enable_seg=False,
                 smooth_seg=True, det_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.complexity = complexity
        self.landmarks = landmarks
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.det_conf = det_conf
        self.track_conf = track_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.landmarks,
                                     self.enable_seg, self.smooth_seg, self.det_conf,
                                     self.track_conf)

    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if draw:
            if self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmlist


def calculateAngle(landmark1, landmark2, landmark3):
    _, x1, y1 = landmark1
    _, x2, y2 = landmark2
    _, x3, y3 = landmark3
    angle1 = math.atan2(y3 - y2, x3 - x2)
    angle2 = math.atan2(y1 - y2, x1 - x2)
    angle = math.degrees(angle1 - angle2)
    if angle < 0:
        angle += 360

    return angle


def process_angle(img, yy, lm1, lm2, lm3, ref_angle):
    angle = calculateAngle(lm1, lm2, lm3)
    # 각도차이를 구한다.
    diff = angle - ref_angle
    if diff < 0: diff += 360
    if diff > 180: diff = 360 - diff
    success = diff < 20

    # 각도가 20도 이하이면 성공 파란색, 틀리면 붉은색
    color = (255, 0, 0) if success else (0, 0, 255)
    # cv2.circle(img, (lm1[1], lm1[2]), 10, (0, 0, 255), 2)
    cv2.circle(img, (lm2[1], lm2[2]), 10, color, 2)
    # cv2.circle(img, (lm3[1], lm3[2]), 10, (0, 0, 255), 2)
    # cv2.putText(img, str(int(angle)), (20, yy), cv2.FONT_HERSHEY_PLAIN, 2, color, 1)
    # cv2.putText(img, str(int(ref_angle)), (100, yy), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)
    # cv2.putText(img, str(int(diff)), (180, yy), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)
    return success


def main():
    cap = cv2.VideoCapture(1)
    cap.set(3, 480)
    cap.set(4, 640)
    pTime = 0
    detector = poseDetector()
    image = cv2.imread("TREE-M.png")
    # Trackbar(image)
    Trackbar()
    while True:
        _, img = cap.read()
        img = detector.findPose(img)
        lmlist = detector.findPosition(img)
        # print(lmlist)
        if len(lmlist) > 0:

            # 24-26-28: 180
            # 23-25-27: 270
            # 11-23-27: 180
            # 12-24-28: 180
            # 11-13-15: 315
            # 12-14-16: 45
            # 16-20-15: 300
            # 16-19-15: 300
            # 24-12-14: 45
            # 23-11-13: 315
            # 13-15-19: 240
            # 12-14-16: 45
            seccess = 0
            y = 0
            dy = 30
            y += dy
            if process_angle(img, y, lmlist[24], lmlist[26], lmlist[28], 180): seccess += 1
            y += dy
            if process_angle(img, y, lmlist[23], lmlist[25], lmlist[27], 270): seccess += 1
            y += dy
            if process_angle(img, y, lmlist[11], lmlist[23], lmlist[27], 180): seccess += 1
            y += dy
            if process_angle(img, y, lmlist[12], lmlist[24], lmlist[28], 180): seccess += 1
            y += dy
            if process_angle(img, y, lmlist[11], lmlist[13], lmlist[15], 315): seccess += 1
            y += dy
            if process_angle(img, y, lmlist[12], lmlist[14], lmlist[16], 45): seccess += 1
            y += dy
            if process_angle(img, y, lmlist[16], lmlist[20], lmlist[15], 300): seccess += 1
            y += dy
            if process_angle(img, y, lmlist[16], lmlist[19], lmlist[15], 300): seccess += 1
            y += dy
            if process_angle(img, y, lmlist[24], lmlist[12], lmlist[14], 45): seccess += 1
            y += dy
            if process_angle(img, y, lmlist[23], lmlist[11], lmlist[13], 315): seccess += 1
            y += dy
            if process_angle(img, y, lmlist[13], lmlist[15], lmlist[19], 240): seccess += 1
            y += dy
            if process_angle(img, y, lmlist[12], lmlist[14], lmlist[16], 45): seccess += 1
            def ten():
                in_sec = 10
                sec = int(in_sec)
                print(sec)

                # while은 반복문으로 sec가 0이 되면 반복을 멈춰라
                while (sec != 0):
                    sec = sec - 1
                    time.sleep(1)
                    print(sec)
            # if seccess == 12:
            #     ten()


            # print(lmlist[11])
            # print(lmlist[12])
            # #print(lmlist[13])
            # angle = calculateAngle(lmlist[12], lmlist[11], lmlist[13])
            # # print(angle)
            # cv2.circle(img, (lmlist[11][1], lmlist[11][2]), 10, (0, 0, 255), 2)
            # cv2.circle(img, (lmlist[12][1], lmlist[12][2]), 10, (0, 0, 255), 2)
            # cv2.circle(img, (lmlist[13][1], lmlist[13][2]), 10, (0, 0, 255), 2)

            # cv2.putText(img, str(int(angle)), (70, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 1)


        # FPS계산 및 화면 표시
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (300, 50), cv2.FONT_HERSHEY_PLAIN, 3, (205, 30, 140), 3)

        cv2.imshow("Pose Estimation", np.hstack((img, ref_img)))

        key = cv2.waitKey(27)
        if key == 27:  # ESC를 누르면 무한루프를 빠져나오게 한다.
            break

    # 프로그램 종료. 카메라 리소스를 해제하고, 모든 창을 닫습니다.
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
