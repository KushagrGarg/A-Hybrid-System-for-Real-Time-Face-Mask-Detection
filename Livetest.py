from keras.models import load_model
import cv2
import numpy as np
import pyaudio
#from playsound import playsound
import wave

def beep():
    p = pyaudio.PyAudio()
    volume = 0.9    
    fs = 44100       
    duration = 1.0   
    f = 440.0        
    samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
    stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)
    stream.write(volume*samples)

    stream.stop_stream()
    stream.close()

    p.terminate()

class LiveTest:
    def __init__(self):
        self.livemodel = load_model('model-012.model/')
        self.face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.source=cv2.VideoCapture(0)
        
    def Test(self):
        labels_dict={0:'MASK',1:'NO MASK'}
        color_dict={0:(0,255,0),1:(0,0,255)}
        while(True):
            ret,img=self.source.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=self.face_clsfr.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                face_img=gray[y:y+w,x:x+w]
                resized=cv2.resize(face_img,(100,100))
                normalized=resized/255.0
                reshaped=np.reshape(normalized,(1,100,100,1))
                result=self.livemodel.predict(reshaped)
                label=np.argmax(result,axis=1)[0]
                if label==0:
                    beep()
                cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
                cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
                cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.imshow('LIVE',img)
            if cv2.waitKey(1)  & 0xFF == ord('q'):
                break
            #key=cv2.waitKey(1)
    
            #if(key==27):
                #break
        
        cv2.destroyAllWindows()
        self.source.release()