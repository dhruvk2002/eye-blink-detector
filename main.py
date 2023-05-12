import cv2 #for video capture and rendering
import dlib  #for face detection
from imutils import face_utils #for converting face features to numpy arrays
import numpy as np #for mathematical operations
import time #for time related functions
from scipy.spatial import distance as dist
# PATH=input('Input the image path with single quotes: ')
ptime=0
cam = cv2.VideoCapture(0)
#----Variables Needed----#
count=0
end_frame=3
thresh=0.4
 #-------#
detector= dlib.get_frontal_face_detector()
our_model=dlib.shape_predictor('DATASETS/shape_predictor_68_face_landmarks.dat')
#----Eye Landmarks----#
(L_start,L_end)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start,R_end)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def ret_EAR(eye):
 #----Verticle Lines----#
 v1=dist.euclidean(eye[1],eye[5])
 v2=dist.euclidean(eye[2],eye[4])

 #----Horizontal Lines----#
 h1=dist.euclidean(eye[0],eye[3])

 #----Calculation----#
 ear=(v1+v2)/h1
 return ear

while True:
 _ , frame= cam.read()
 # Code for resizing and looping of video if it is not live
 #if cam.get(cv2.CAP_PROP_POS_FRAMES)== cam.get(cv2.CAP_PROP_FRAME_COUNT)
 # cam.set(cv2.CAP_PROP_POS_FRAMES,0)
 #frame= cv2.resize(frame(720,640))
 #-------FPS-------#
 ctime= time.time()
 fps=1/(ctime-ptime)
 ptime=ctime
 cv2.putText(
  frame, 
   F'Current Frame Rate:{int(fps)}',
   (50,50), 
   cv2.FONT_HERSHEY_DUPLEX,
   1,
   (0,100,0),
   1
 )
 gray_frm=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 faces=detector(gray_frm)
 for face in faces:
  x1=face.left()
  y1=face.top()
  x2=face.right()
  y2=face.bottom()
  cv2.rectangle(frame,(x1,y1),(x2,y2),(200),2)

  #----Landmarks----#
  shapes=our_model(gray_frm,face)
  shape= face_utils.shape_to_np(shapes)
  #print(shape)
  #print(shape.shape)
  #---Eye---#
  lefteye=shape[L_start:L_end]
  righteye=shape[R_start:R_end]
 
#----For Marking Eye points----#
  for Lpt,Rpt in zip(lefteye,righteye):
   cv2.circle(frame,Lpt,2,(200,200,0),2)
   cv2.circle(frame,Rpt,2,(200,200,0),2)

  #----EAR----#
  left_ear=ret_EAR(lefteye)
  right_ear=ret_EAR(righteye)
  avg=(left_ear+right_ear)/2
  # print(avg)
  if avg<thresh:
   count+=1
  else:
   if count>end_frame:
    cv2.putText(frame,f'BLINK DETECTED',(frame.shape[1]//2 -200,frame.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,2,(0,200,0),2)
   else:
    c=0

 cv2.imshow('Captured Content',frame)
 if cv2.waitKey(3) & 0xFF== ord('x'):
  break
cam.release()
