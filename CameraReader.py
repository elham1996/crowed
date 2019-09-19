"""
auther @Eng.Elham albaroudi
"""

from VideoGet import VideoGet
from VideoShow import VideoShow
from SaveImage import SaveImage


def threadBoth(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(source).start()
    video_shower = VideoShow(video_getter.frame).start()
    save_image = SaveImage(video_getter.frame)
    save_image.start()
    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            save_image.stop()
            #save_image.join()
            break

        frame = video_getter.frame
        video_shower.frame = frame
        save_image.frame = frame

if __name__ == '__main__':
    #source ="rtsp://admin:12345678a@192.168.100.219:554/ISAPI/streaming/channels/101"
    source= "My Video.mp4"
    threadBoth(source)
