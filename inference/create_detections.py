"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm
import argparse
from det_util import generate_detections
import boto3
import io
import os
import time
import sys

"""
Inference script to generate a file of predictions given an input.

Args:
    checkpoint: A filepath to the exported pb (model) file.
        ie ("saved_model.pb")

    chip_size: An integer describing how large chips of test image should be

    input: A filepath to a single test chip
        ie ("1192.tif")

    output: A filepath where the script will save  its predictions
        ie ("predictions.txt")


Outputs:
    Writes a file specified by the 'output' parameter containing predictions for the model.
        Per-line format:  xmin ymin xmax ymax class_prediction score_prediction
        Note that the variable "num_preds" is dependent on the trained model
        (default is 250, but other models have differing numbers of predictions)

"""


def chip_image(img, chip_size=(300, 300)):
    """
    Segment an image into NxWxH chips

    Args:
        img : Array of image to be chipped
        chip_size : A list of (width,height) dimensions for chips

    Outputs:
        An ndarray of shape (N,W,H,3) where N is the number of chips,
            W is the width per chip, and H is the height per chip.

    """
    width, height, _ = img.shape
    wn, hn = chip_size
    images = np.zeros((int(width/wn) * int(height/hn), wn, hn, 3))
    k = 0
    for i in tqdm(range(int(width/wn))):
        for j in range(int(height/hn)):

            chip = img[wn*i:wn*(i+1), hn*j:hn*(j+1), :3]
            images[k] = chip

            k = k + 1

    return images.astype(np.uint8)


def draw_bboxes(img, boxes, classes):
    """
    Draw bounding boxes on top of an image

    Args:
        img : Array of image to be modified
        boxes: An (N,4) array of boxes to draw, where N is the number of boxes.
        classes: An (N,1) array of classes corresponding to each bounding box.

    Outputs:
        An array of the same shape as 'img' with bounding boxes
            and classes drawn

    """
    source = Image.fromarray(img)
    draw = ImageDraw.Draw(source)
    w2, h2 = (img.shape[0], img.shape[1])

    idx = 0

    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i]
        c = classes[i]

        draw.text((xmin+15, ymin+15), str(c))

        for j in range(4):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
    return source

def processChip(data):
    global chipStart, chipEnd, detectionStart, detectionEnd, rescaleStart, rescaleEnd

    # Capture chip start timing
    chipStart = time.time()

    # Parse and chip images
    img = Image.open(data) if type(data) is str else Image.open(io.BytesIO(data))
    arr = np.array(img)
    chip_size = (args.chip_size, args.chip_size)
    images = chip_image(arr, chip_size)
    print("\n" + str(images.shape))
    sys.stdout.flush()

    # Capture timing
    chipEnd = time.time()

    # generate detections
    detectionStart = time.time()
    boxes, scores, classes = generate_detections(
        detection_graph, images)
    detectionEnd = time.time()

    rescaleStart = time.time()

    # Process boxes to be full-sized
    width, height, _ = arr.shape
    cwn, chn = (chip_size)
    wn, hn = (int(width/cwn), int(height/chn))

    num_preds = 250
    bfull = boxes[:wn*hn].reshape((wn, hn, num_preds, 4))
    b2 = np.zeros(bfull.shape)
    b2[:, :, :, 0] = bfull[:, :, :, 1]
    b2[:, :, :, 1] = bfull[:, :, :, 0]
    b2[:, :, :, 2] = bfull[:, :, :, 3]
    b2[:, :, :, 3] = bfull[:, :, :, 2]

    bfull = b2
    bfull[:, :, :, 0] *= cwn
    bfull[:, :, :, 2] *= cwn
    bfull[:, :, :, 1] *= chn
    bfull[:, :, :, 3] *= chn
    for i in range(wn):
        for j in range(hn):
            bfull[i, j, :, 0] += j*cwn
            bfull[i, j, :, 2] += j*cwn

            bfull[i, j, :, 1] += i*chn
            bfull[i, j, :, 3] += i*chn

    bfull = bfull.reshape((hn*wn, num_preds, 4))

    rescaleEnd = time.time()

    #only display boxes with confidence > .5
    if type(data) is str:
        bs = bfull[scores > .5]
        cs = classes[scores>.5]
        s = data.split("\\")[::-1]
        os.makedirs("bboxes", exist_ok=True)
        draw_bboxes(arr,bs,cs).save("bboxes\\"+s[0].split(".")[0] + ".png")

    with open(basename + '.predictions.txt', 'w') as f:
        for i in range(bfull.shape[0]):
            for j in range(bfull[i].shape[0]):
                # box should be xmin ymin xmax ymax
                box = bfull[i, j]
                class_prediction = classes[i, j]
                score_prediction = scores[i, j]
                f.write('%d %d %d %d %d %f \n' %
                        (box[0], box[1], box[2], box[3], int(class_prediction), score_prediction))


if __name__ == "__main__":
    global chipStart, chipEnd, detectionStart, detectionEnd


    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--access_key_id", help="AWS Access key id")
    parser.add_argument('-s', "--secret_key", help="AWS Secret Key")
    parser.add_argument('-b', "--bucket", help="AWS Bucket")
    parser.add_argument('-p', "--prefix", help="Prefix filter")
    parser.add_argument('-r', "--region", help="Region Identifier")
    parser.add_argument('-l', "--log", help="timing log")
    parser.add_argument("-c", "--checkpoint",
                        default='pbs/model.pb', help="Path to saved model")
    parser.add_argument("-cs", "--chip_size", default=300,
                        type=int, help="Size in pixels to chip input image")
    parser.add_argument("-d", "--dir", help="Path to test chip directory")

    #parser.add_argument("-o","--output",default="predictions.txt",help="Filepath of desired output")
    args = parser.parse_args()

    print("Creating Graph...")
    sys.stdout.flush()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    boxes = []
    scores = []
    classes = []
    k = 0

    if (not args.dir):
        # Establish aws session
        session = boto3.Session(
            aws_access_key_id=args.access_key_id,
            aws_secret_access_key=args.secret_key,
            region_name=args.region
        )

        s3 = session.resource('s3')

        bucket = s3.Bucket(args.bucket)

    l = open(args.log, "a")
    l.write(
        "time, key, download_time, chip_time, inference_time, rescale_time, complete_time\n")
    l.close()

    if (not args.dir):
        for o in bucket.objects.filter(Prefix=args.prefix):
            basename = os.path.basename(o.key)
            if o.key.endswith(".tif") and not (os.path.exists(basename + '.predictions.txt')):

                completeStart = time.time()

                # Download content to memory object
                downloadStart = time.time()
                response = o.get()
                data = response['Body'].read()
                downloadEnd = time.time()

                processChip(data)

                completeEnd = time.time()

                key = o.key
                download_time = downloadEnd - downloadStart
                chip_time = chipEnd - chipStart
                inference_time = detectionEnd - detectionStart
                rescale_time = rescaleEnd - rescaleStart
                complete_time = completeEnd - completeStart

                l = open(args.log, "a")
                l.write(",".join(list(map(lambda x: str(x), [time.time(),
                        key, download_time, chip_time, inference_time, rescale_time, complete_time]))) + "\n")
                l.close()
    else:
        for f in os.listdir(args.dir):
            basename = os.path.basename(f)
            if f.endswith(".tif") and not (os.path.exists(basename + '.predictions.txt')):

                completeStart = time.time()
                processChip(os.path.join(args.dir, f))
                completeEnd = time.time()

                key = f
                chip_time = chipEnd - chipStart
                inference_time = detectionEnd - detectionStart
                rescale_time = rescaleEnd - rescaleStart
                complete_time = completeEnd - completeStart

                l = open(args.log, "a")
                l.write(",".join(list(map(lambda x: str(x), [time.time(),
                        key, 0, chip_time, inference_time, rescale_time, complete_time]))) + "\n")
                l.close()
