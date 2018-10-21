from __future__ import print_function

import json
import os
import sys
import json
import ConfigParser
import boto3

#pull in the vendored directory because it will contain all the third party libraries
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(HERE, "vendor"))

#pull in the third party libraries needed for this example
from PIL import Image
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf

from object_detection.utils import ops as utils_ops

s3 = boto3.client('s3')

#pull in the model from s3
s3.download_file('mlstuff7631', 'model/frozen_inference_graph.pb', '/tmp/frozen_inference_graph.pb')

#load model from the local file
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile('/tmp/frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#perform object detection on a single image
def run_inference_for_single_image(image, graph):
    with graph.as_default():
      with tf.Session() as sess:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
        if 'detection_masks' in tensor_dict:
          # The following processing is only for single image
          detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
          detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
          # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
          real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
          detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
          detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
          detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              detection_masks, detection_boxes, image.shape[0], image.shape[1])
          detection_masks_reframed = tf.cast(
              tf.greater(detection_masks_reframed, 0.5), tf.uint8)
          # Follow the convention by adding back the batch dimension
          tensor_dict['detection_masks'] = tf.expand_dims(
              detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
  
        # Run inference
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: np.expand_dims(image, 0)})
  
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
          output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

#calculate the boundary box area
def boxArea(box):
    return (box[2]-box[0])*(box[3]-box[1])

#determine if two boundary boxes intersect
def doBoxesIntersect(box1, box2):
    if ((box1[1] < box2[3]) 
        & (box1[3] > box2[1]) 
        & (box1[0] < box2[1]) 
        & (box1[1] > box2[0])):
        return True
    else:
        return False
    
#combine two boundary boxes
def combineBoxes(box1, box2):
    return[np.fmin(box1[0],box2[0]),
           np.fmin(box1[1],box2[1]),
           np.fmax(box1[2],box2[2]),
           np.fmax(box1[3],box2[3])]
    
MARGIN_RATIO = 1700

#crop a single image by centering a person and a bike
def processAndCropImage(image_path, output_dict):
    img = Image.open(image_path)
    width = img.size[0]
    height = img.size[1]
    
    
    # find person box
    personBox = [0,0,0,0]
    for index, currentBox in enumerate(output_dict['detection_boxes']):
        if((boxArea(personBox) < boxArea(currentBox)) 
           and (output_dict['detection_scores'][index] > 0.5) 
           and (output_dict['detection_classes'][index] == 1)):
            personBox = currentBox
    
    # find intersecting bike box
    bikeBox = [0,0,0,0]
    for index, currentBox in enumerate(output_dict['detection_boxes']):
        if(doBoxesIntersect(currentBox, personBox) 
           and (output_dict['detection_scores'][index] > 0.5) 
           and (output_dict['detection_classes'][index] == 2) 
           and (boxArea(bikeBox) < boxArea(currentBox))):
            bikeBox = currentBox
    
    # combine boxes if there is an overlaping bike
    if(bikeBox[0] !=0 ):
        personBox = combineBoxes(bikeBox, personBox)
    
    # fix aspect ratio to make sure all images have the same
    
    # desired ratio: 2.7847145
    diff = 2.7847145 * (personBox[3] - personBox[1]) - (personBox[2] - personBox[0])
    personBox[2] = personBox[2]+diff/2
    personBox[0] = personBox[0]-diff/2
    
    margin = (personBox[3] - personBox[1])*MARGIN_RATIO
    
    area = (width * personBox[1] - margin, 
            height * personBox[0] - margin, 
            width * personBox[3] + margin, 
            height * personBox[2] + margin)
    
    cropped_img = img.crop(area)
    
    return cropped_img

    
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

#lambda function
def lambda_handler(event, context):
    
    for record in event["Records"]:
      
      #get the file name from the key that was provided by the S3 event
      split_key = record["s3"]["object"]["key"].split('/')
      file_name = split_key[len(split_key)-1]
      
      #get the file from S3 and store it locally
      s3.download_file(record["s3"]["bucket"]["name"], record["s3"]["object"]["key"], '/tmp/input_'+file_name)
      
      image = Image.open('/tmp/input_'+file_name)
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      
      #identify the objects in the image and get their boundaries
      output_dict = run_inference_for_single_image(image_np, detection_graph)
      
      #crop the image based on the centered object(s)
      cropped_img = processAndCropImage('/tmp/input_'+file_name, output_dict)
      
      #save locally and then save to S3
      cropped_img.save('/tmp/output_'+file_name)
      s3.upload_file('/tmp/output_'+file_name, record["s3"]["bucket"]["name"], 'output_images/'+file_name)

    return 'Image Saved to: '+'output_images/'+file_name+' in'+record["s3"]["bucket"]["name"]
