import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
cap=cv2.VideoCapture("stock-footage-close-up-head-shot-portrait-outdoors-s-beautiful-calm-serious-sad-upset-woman-look-at-camera.webm")

idList=[22,23,24,26,110,157,158,159,160,161,130,243]
detector=FaceMeshDetector(maxFaces=1)
plotY=LivePlot(640,360,[40,120],invert=True)
ratioList=[]
blinkCounter=0
counter=0
color= (255,0,255)
while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    
    success,img=cap.read()
    img,faces=detector.findFaceMesh(img,draw=False)
    if faces:
        face=faces[0]
        for id in idList:
            cv2.circle(img,face[id],5,color,cv2.FILLED)
        leftUp=face[159]
        leftDown=face[23]
        leftLeft=face[130]
        leftRight=face[243]
        lengthVer,_=detector.findDistance(leftUp,leftDown)
        lengthHor,_=detector.findDistance(leftLeft,leftDown)
        cv2.line(img,leftUp,leftDown,(0,200,0),3)
        cv2.line(img,leftLeft,leftRight,(0,200,0),3)
        ratio=((lengthVer/lengthHor)*100)
        ratioList.append(ratio)
        if len(ratioList)>5:
            ratioList.pop(0)
        ratioAvg=sum(ratioList)/len(ratioList)

        if ratioAvg<80 and counter == 0:
            blinkCounter += 1
            color=(0,200,0)
            counter=1
        if counter != 0:
            counter +=1
            if counter>10:
                counter =0
                color=(255,0,255)

        cvzone.putTextRect(img,f'Blink Count:{blinkCounter}',(50,50))

        imgPlot=plotY.update(ratioAvg,color)
        img=cv2.resize(img,(640,360))
        # cv2.imshow("ImagePlot",imgPlot)
        imageStack=cvzone.stackImages([img,imgPlot],1,1)
    else:
        img=cv2.resize(img,(640,320))
        # cv2.imshow("ImagePlot",imgPlot)
        imageStack=cvzone.stackImages([img,img],1,1)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray = cv2.bilateralFilter(gray,5,1,1)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(200,200))

    
   
    cv2.imshow("Image",imageStack)
    cv2.waitKey(55)