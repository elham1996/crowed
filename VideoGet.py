"""
auther @Eng.Elham albaroudi

"""

from threading import Thread
import cv2
import time
class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        capture = cv2.VideoCapture(src)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
        self.stream = capture
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
  
    def stop(self):
        self.stopped = True
        self.stream.release()
        cv2.destroyAllWindows()