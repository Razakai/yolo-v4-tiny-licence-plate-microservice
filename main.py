import os
import pika
import sys
from detect_video import detect_video
import json


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='testingDetection')

    def callback(ch, method, properties, body):
        msg = json.loads(body)["message"]
        print(" [x] detection Received", msg)
        detect_video(msg)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue='testingDetection', on_message_callback=callback)

    channel.basic_qos(prefetch_count=1)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)