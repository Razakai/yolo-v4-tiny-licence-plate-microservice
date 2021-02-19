# import thread
import time
import tensorflow as tf
import threading
from concurrent.futures import ThreadPoolExecutor

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

'''
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
'''

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
input_size = 416

saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4Tiny-416', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

'''
class myThread(threading.Thread):
    def __init__(self, threadID, url):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.url = url

    def run(self):
        main(self.threadID, self.url)
'''


def main(id, return_value, frame):
    """
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416

    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4Tiny-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    """

    #vid = cv2.VideoCapture(url)

    # out = None

    currentFrame = 0
    #while True:
    #return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
    else:
        print('Video has ended or failed, try a different video format!')
        return None

    currentFrame += 1
    if (currentFrame % 4) == 0:
        #continue
        return None

    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    start_time = time.time()

    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(frame, pred_bbox)
    fps = 1.0 / (time.time() - start_time)
    print(id, "FPS: %.2f" % fps)
    result = np.asarray(image)
    cv2.namedWindow(str(id), cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imshow(str(id), result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return None
    #cv2.destroyAllWindows()

#id, return_value, frame

def task():
    executor = ThreadPoolExecutor(max_workers=6)
    cap1 = cv2.VideoCapture('http://192.168.1.103:1234/video')
    cap2 = cv2.VideoCapture('http://192.168.1.103:1234/video')
    currentFrame = 0
    while True:
        currentFrame += 1
        if (currentFrame % 3) == 0:
            continue
        return_value1, frame1 = cap1.read()
        return_value2, frame2 = cap2.read()
        print('start')
        executor.submit(main(1, return_value1, frame1))
        print('stop')
        executor.submit(main(2, return_value2, frame2))

#  python detect_video.py --weights ./checkpoints/yolov4Tiny-416 --size 416 --model yolov4 --tiny

if __name__ == '__main__':
    try:
        task()
        # app.run(main)
        # main('http://192.168.1.103:1234/video')
        #thread1 = myThread(1, 'http://192.168.1.103:1234/video')
        #thread2 = myThread(2, 'http://192.168.1.103:1234/video')
        #thread1.start()

        #print('1')

        #thread2.start()

        #print('2')

    except SystemExit:
        pass
