"""
 
"""

from threading import Thread
import time
from model import CSRNet
import torch
from image import *
import PIL.Image as Image
import datetime
import tensorflow as tf
min_threshold = 0.5

from torchvision import datasets, transforms
transform=transforms.Compose([
                      transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                  ])

model = CSRNet()
model.cuda()
graph = tf.get_default_graph()

     

class SaveImage(Thread):
    """
    Class used to store image
    """

    def __init__(self, frame=None):
        Thread.__init__(self)
        self.frame = frame
        self.stopped = False
        self.img_counter =0


    def run(self):
         while not self.stopped:
             img_name = "opencv_frame_{}.png".format(self.img_counter)
             cv2.imwrite(img_name, self.frame)
             print("{} written!".format(img_name))
             self.img_counter
             global graph   
             with graph.as_default():
                     image = cv2.imread(img_name)
                     image = cv2.resize(image, (416, 416))
                     checkpoint = torch.load('PartAmodel_best.pth.tar')
                     model.load_state_dict(checkpoint['state_dict'])
                     img = transform(Image.fromarray(image).convert('RGB')).cuda()
                     output = model(img.unsqueeze(0))
                     timess = str(datetime.datetime.now())

                     print("Predicted Count : ", int(output.detach().cpu().sum().numpy())+10)
                     print('time: ', timess)



    time.sleep(5)

    def stop(self):
        self.stopped = True
