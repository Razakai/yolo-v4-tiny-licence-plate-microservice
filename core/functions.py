import pika
import json
import cv2
import numpy as np
from PIL import Image

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='baka')

def crop_detections(img, data, width, height):
    boxes, scores, classes, num_objects = data
    #print(boxes, scores, classes, num_objects)
    detectionArr = []
    print('num detections', num_objects)
    for detection in range(num_objects):
        xmin, ymin, xmax, ymax = boxes[detection]

        cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
        # convert to numpy array
        cropped_img = np.array(cropped_img)
        # resize to specified height and width
        cropped_img = cv2.resize(cropped_img, (width, height), interpolation=cv2.INTER_AREA)
        # greyscale image
        cropped_img = (cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) / 127.5) - 1.0
        cropped_img = np.expand_dims(cropped_img, axis=2)
        cropped_img = np.squeeze(cropped_img, axis=-1)
        # cropped_img = Image.fromarray(np.uint8((cropped_img + 1.0) * 127.5))
        print(len(cropped_img), cropped_img.shape)
        detectionArr.append(cropped_img.tolist())

    if len(detectionArr) > 0:
        send_images(detectionArr)


def send_images(detections):
    #for item in detections:
    channel.basic_publish(exchange='',
                          routing_key='baka',
                          body=json.dumps({"detections": str(detections)}))


