"""

auther @Eng.Elham albaroudi
"""


from threading import Thread
import time


from model import CSRNet
import torch
from image import *

min_threshold = 0.5

from torchvision import datasets, transforms
transform=transforms.Compose([
                      transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                  ])

model = CSRNet()
model.cuda()
import tensorflow as tf 
import datetime


min_threshold = 0.5

graph = tf.get_default_graph()


from PIL import Image
import matplotlib.pyplot as plt

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
       global graph   
       with graph.as_default():  
         while not self.stopped:
            self.frame = cv2.resize(self.frame, (416, 416))
            checkpoint = torch.load('PartAmodel_best.pth.tar')
            model.load_state_dict(checkpoint['state_dict'])
            img = transform(Image.fromarray(self.frame).convert('RGB')).cuda()
            output = model(img.unsqueeze(0))
            timess = str(datetime.datetime.now())
            print("Predicted Count : ", int(output.detach().cpu().sum().numpy())+10)
            print('time: ', timess)
            cv2.imshow("Video", self.frame)
            k = cv2.waitKey(1)
            if k%256 == 27:
               # ESC pressed
               self.stopped = True

    def stop(self):
        self.stopped = True
