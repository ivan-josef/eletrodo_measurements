import cv2 
import numpy as np 
import os

class Sub_Pixel_Edge():
    def __init__(self,path):
        self.path = path
        self.files = os.listdir(self.path)
        self.index = 0 

    def run(self):
        while True:
            self.complete_path = os.path.join(self.path,self.files[self.index])
            self.img_raw = cv2.imread(self.complete_path)
            self.img_raw = cv2.resize(self.img_raw,(1920,1080))
            cv2.imshow('eletrodo',self.img_raw)

            self.key = cv2.waitKey(0) & 0xFF 
            if self.key == ord('d'):
                self.index += 1
            elif self.key == ord('a'):
                self.index -= 1
            elif self.key == ord('q'):
                break

            if self.index < 0:
                self.index = 0
            if self.index >= len(self.files) - 1:
                self.index = len(self.files) - 1


path = 'images'
obj = Sub_Pixel_Edge(path)
obj.run()
cv2.destroyAllWindows()
            