import cv2
import tensorflow as tf
import numpy as np
import os 

sys.path.appand('..')

from utils import label_map_util
from utils import visualization_utils as vis_util

path_ckpt = 'ssd_mobilenet_v1_coco_2017_11_17' + '/frozen_inference_graph.pb'
path_label = os.path.join('data', 'mscoco_label_map.pbtxt')

## have to know what os.join means

NUM_CLASSES = 90 ## what is it?

img = None

#####지정된 url로 다운로드를 진행하고 알집을 풀고 그 경로를 받아서 os의 기본 네임으로 설정
#####뭐 대강 확인



## 아마도 모델을 메모리로 불러오는 과정 인듯
detection_graph = tf.Graph()
with detection_graph.as_default() :
	od_graph_def = tf.GraphDef ()
	with tf.gfile.GFile(path_ckpt, 'rb') as fid :
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(ob_graph_def, name = '')


label_map = label_map_util.load_labelmap(path_label)
categories = label_map_util.convert_label_map_to_categories(labelmap, max_num_classes = NUM_CLASSES, use_display_name = True)
category_index = label_map_util.create_category_index(categories)

def get_img (frame):
	global img

	img = frame

def process_img() :
	global img

	if img is None :
		print('no_img')
		return

	with detection_graph.as_default() :
		with tf.Session(graph = detection_graph) as sess :
			img_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_scores = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			img_ex = np.expand_dims(img, axis = 0)

			(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict ={img_tensor : img_ex})

			vis_util.visualize_boxes_and_labels_on_image_array(img, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness = 8)

			obstalces = list()

			for box in boxes : 
				obstalces.append('row' : box[0], 'col' : box[1], 'width' : box[2], 'height' : box[3]) ###MAY BE

			return img, boxes


