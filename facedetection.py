import cv2
import mediapipe as mp
import pafy
import time

url = "https://www.youtube.com/watch?v=9LCh3j_yxFQ&ab_channel=MidasShrestha"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture()
cap.open(best.url)
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):

            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame,bbox,(253, 253, 0),5)
            cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (50, 205, 50), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,f'FPS: {int(fps)}',(255,71), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Output",frame)
    k = cv2.waitKey(6)



cap.release()
cv2.destroyAllWindows()
