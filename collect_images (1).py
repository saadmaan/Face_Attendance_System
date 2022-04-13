import cv2
import os
import face_recognition
import time

c =0
name = 'noy'
# os.chdir('/home/genex/jnotebook')
# os.mkdir(f'./train_dirNew/test')
# dir = f'./train_dirNew/test/'

def collect(dir):
    c=0
    vs = cv2.VideoCapture("rtsp://admin:Genex@321@103.85.159.78:554/Streaming/Channels/201")
    # os.chdir(dir)
    time.sleep(3)
    name = dir.split('_')[-1]
    # name = "ddd"
    while True:
        ret, frame = vs.read()
        if ret:
            wframe = frame.copy()
            width = int(wframe.shape[1] * .25)
            height = int(wframe.shape[0] * .25)
            dim = (width, height)
            wframe = cv2.resize(wframe, dim, interpolation = cv2.INTER_AREA)
            wframe = cv2.cvtColor(wframe, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(wframe)
            if len(faces) == 1:
                print("???????????????>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<???????????")
                faces[0] = faces[0][0]*4,faces[0][1]*4,faces[0][2]*4,faces[0][3]*4    
                y1,x2,y2,x1=faces[0]
                per = frame[y1-20:y2+40,x1-10:x2+10]
#                 ratio = ((y2-y1)/frame.shape[0])*100
#                 if ratio > 20: 
                c = c+1
#                     width = int(frame.shape[1] * 4)
#                     height = int(frame.shape[0] * 4)
#                     dim = (width, height)
#                     frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#                     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                filename = 'aa'+str(c)+'.jpeg'
                dir = f'./train_dirNew/445_saad'
                cv2.imwrite(f'{dir}/{filename}', per)
                # cv2.imwrite(filename, per)
                # if c %  == 0:
                #     print("Move Position")
                #     time.sleep(2)
                if c >= 40:
                    break
                    
if __name__ == '__main__':
    collect(dir)          
                    
    