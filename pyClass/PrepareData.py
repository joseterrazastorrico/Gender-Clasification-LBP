import os
import math
import random
import pandas as pd
from skimage.transform import rotate
from tqdm import tqdm
import cv2

class PrepareData:
    def __init__(self, dir_input, dir_output, cats=[], Train={}, Test={}):
        self.dir_input = dir_input
        os.makedirs(dir_output, exist_ok=True)
        self.dir_output = dir_output
        self.cats = cats
        self.Train = Train
        self.Test = Test
    
    def TrainTestSplit(self, p=0.7):
        os.makedirs(os.path.join(self.dir_output,"Train"), exist_ok=True)
        os.makedirs(os.path.join(self.dir_output,"Test"), exist_ok=True)
        for i in os.listdir(self.dir_input):
            self.cats.append(i)
            os.makedirs(os.path.join(self.dir_output,"Train",i), exist_ok=True)
            os.makedirs(os.path.join(self.dir_output,"Test",i), exist_ok=True)
            ## Leer archivos de la clase i
            class_i = os.listdir(os.path.join(self.dir_input,i))
            ## Generar el tamaño de la muestra
            sampleSize = math.floor(p*len(class_i))
            ## Seleccionar archivos de entrenamiento y testeo de la clase i
            self.Train[i] = random.sample(class_i, sampleSize)
            self.Test[i] = list(set(class_i)-set(self.Train[i]))
            
    def CROP_SAVE(self,img, dir, name, x, y, w, h, p):
        img2 = img[y-p+1:y+h+p, x-p+1:x+w+p]
        dim = (100,100)
        img2_resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(dir,name), img2_resized)
        
    def DataAugmentation(self,lista, dir_input, dir_output, rots=[-30,-15,15,30], p = 0):
        j = 0
        for i in tqdm(range(len(lista)), desc='Progreso {}'.format(" ".join(dir_output.split("\\")[-2:]))):
            #tqdm(j, desc="Progreso")
            ## LEER IMAGEN
            img = cv2.imread(os.path.join(dir_input,lista[i]), cv2.IMREAD_COLOR)
            
            ## IDETIFICAR ROSTRO
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            try:
                faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                (x, y, w, h) = faces_detected[0] 
            except:
                faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
                (x, y, w, h) = faces_detected[0]
            
            self.CROP_SAVE(img, dir_output, 'image_{}.png'.format(j), x, y, w, h,p)
            j=j+1           
            ## DATA AUG
            ## GIRAR ROSTRO
            img_mod = cv2.flip(img, 1)
            self.CROP_SAVE(img_mod, dir_output, 'image_{}.png'.format(j), x, y, w, h,p)
            j=j+1          
            ## ROTAR
            rows, cols = img.shape[:2]
            for r in rots:
                M = cv2.getRotationMatrix2D((cols/2, rows/2), r, 1)
                img_rotated = cv2.warpAffine(img, M, (cols,rows))
                self.CROP_SAVE(img_rotated, dir_output, 'image_{}.png'.format(j), x, y, w, h,p)
                j=j+1
    def DataAugmentation_all(self):
        for i in os.listdir(self.dir_input):
            self.DataAugmentation(self.Train[i], os.path.join(self.dir_input,i), os.path.join(self.dir_output, 'Train',i))
            self.DataAugmentation(self.Test[i], os.path.join(self.dir_input,i), os.path.join(self.dir_output, 'Test',i))