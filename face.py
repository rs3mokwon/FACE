import cv2 #OpenCV 영상처리
import sys 
import os

def getName(id):
    return id

#classifier
faceCascade = cv2.CascadeClassifier('C:/Users/507/Desktop/object_detection/haarcascade_frontalface_default.xml')

#video caputure setting
capture = cv2.VideoCapture('http://220.81.195.72:5000') # initialize, # is camera number
capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280) #CAP_PROP_FRAME_WIDTH == 3
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720) #CAP_PROP_FRAME_HEIGHT == 4

#console message

face_id = getName(sys.argv[1])

os.makedirs("C:/Users/507/Desktop/object_detection/dataset/"+str(face_id), exist_ok=True)
os.makedirs("C:/Users/507/Desktop/object_detection/dataset/UNKNOWN/"+str(face_id), exist_ok=True)

a = len(os.listdir('C:/Users/507/Desktop/object_detection/dataset/'+str(face_id))) // 300
a += 1

#print("\n [INFO] Initializing face capture. Look the camera and wait ...")

count = 0 # # of caputre face image
while True: 
    ret, frame = capture.read() #카메라 상태 및 프레임
    # cf. frame = cv2.flip(frame, -1) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #흑백으로
    faces = faceCascade.detectMultiScale(
        gray,#검출하고자 하는 원본이미지
        scaleFactor = 1.2, #검색 윈도우 확대 비율, 1보다 커야 한다
        minNeighbors = 6, #얼굴 사이 최소 간격(픽셀)
        minSize=(20,20) #얼굴 최소 크기. 이것보다 작으면 무시
    )

    #얼굴에 대해 rectangle 출력
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #inputOutputArray, point1 , 2, colorBGR, thickness)
        count += 1
        
        cv2.imwrite("C:/Users/507/Desktop/object_detection/dataset/"+str(face_id)+"/User."+str(face_id)+'.'+str(a)+'.'+str(count)+".jpg",gray[y:y+h, x:x+w])
    cv2.imshow('image',frame)

    #종료 조건
    if cv2.waitKey(1) > 0 : break #키 입력이 있을 때 반복문 종료
    elif count >= 300 : break #100 face sample

capture.release() #메모리 해제
cv2.destroyAllWindows()#모든 윈도우 창 닫기
print("End",end='', flush=True)