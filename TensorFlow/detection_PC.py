# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np

# https://www.tensorflow.org/lite/guide/hosted_models
# http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

label2string = \
{
	0:   "person",
	1:   "bicycle",
	2:   "car",
	3:   "motorcycle",
	4:   "airplane",
	5:   "bus",
	6:   "train",
	7:   "truck",
	8:   "boat",
	9:   "traffic light",
	10:  "fire hydrant",
	12:  "stop sign",
	13:  "parking meter",
	14:  "bench",
	15:  "bird",
	16:  "cat",
	17:  "dog",
	18:  "horse",
	19:  "sheep",
	20:  "cow",
	21:  "elephant",
	22:  "bear",
	23:  "zebra",
	24:  "giraffe",
	26:  "backpack",
	27:  "umbrella",
	30:  "handbag",
	31:  "tie",
	32:  "suitcase",
	33:  "frisbee",
	34:  "skis",
	35:  "snowboard",
	36:  "sports ball",
	37:  "kite",
	38:  "baseball bat",
	39:  "baseball glove",
	40:  "skateboard",
	41:  "surfboard",
	42:  "tennis racket",
	43:  "bottle",
	45:  "wine glass",
	46:  "cup",
	47:  "fork",
	48:  "knife",
	49:  "spoon",
	50:  "bowl",
	51:  "banana",
	52:  "apple",
	53:  "sandwich",
	54:  "orange",
	55:  "broccoli",
	56:  "carrot",
	57:  "hot dog",
	58:  "pizza",
	59:  "donut",
	60:  "cake",
	61:  "chair",
	62:  "couch",
	63:  "potted plant",
	64:  "bed",
	66:  "dining table",
	69:  "toilet",
	71:  "tv",
	72:  "laptop",
	73:  "mouse",
	74:  "remote",
	75:  "keyboard",
	76:  "cell phone",
	77:  "microwave",
	78:  "oven",
	79:  "toaster",
	80:  "sink",
	81:  "refrigerator",
	83:  "book",
	84:  "clock",
	85:  "vase",
	86:  "scissors",
	87:  "teddy bear",
	88:  "hair drier",
	89:  "toothbrush",
}

def detect_from_camera():
	# load model
	interpreter = tf.lite.Interpreter(model_path="detect.tflite")
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	#cap = cv2.VideoCapture('./traffic.mp4') # 0はカメラのデバイス番号
	cap = cv2.VideoCapture(0)
	while True:
		# capture image
		ret, img_org = cap.read()
#		cv2.imshow('image', img_org)
		key = cv2.waitKey(1)
		if key == 27: # ESC
			break

		# prepare input image
		img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (300, 300))
		img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 300, 300, 3)
		img = img.astype(np.uint8)
		
		# set input tensor
		interpreter.set_tensor(input_details[0]['index'], img)

		# run
		interpreter.invoke()

		# get outpu tensor
		boxes = interpreter.get_tensor(output_details[0]['index'])
		labels = interpreter.get_tensor(output_details[1]['index'])
		print("Hello World")
		print(output_details)
		scores = interpreter.get_tensor(output_details[2]['index'])
		num = interpreter.get_tensor(output_details[3]['index'])
	
		for i in range(boxes.shape[1]):
			if scores[0, i] > 0.5:
				box = boxes[0, i, :]
				x0 = int(box[1] * img_org.shape[1])
				y0 = int(box[0] * img_org.shape[0])
				x1 = int(box[3] * img_org.shape[1])
				y1 = int(box[2] * img_org.shape[0])
				box = box.astype(np.int)
				cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
				cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
    			
				cv2.putText(img_org,
						str(label2string[int(labels[0, i])]),
					#    str(int(labels[0, i])),
					   (x0, y0),
					   cv2.FONT_HERSHEY_SIMPLEX,
					   1,
					   (255, 255, 255),
					   2)
	
	#	cv2.imwrite('output.jpg', img_org)
		cv2.imshow('image', img_org)
		
	cap.release()
	cv2.destroyAllWindows()


def detect_from_image():
	# prepara input image
	img_org = cv2.imread('./Caesar.jpeg')
#	cv2.imshow('image', img)
	img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (300, 300))
	img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 300, 300, 3)
	img = img.astype(np.uint8)

	# load model
	interpreter = tf.lite.Interpreter(model_path="detect.tflite")
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# set input tensor
	interpreter.set_tensor(input_details[0]['index'], img)

	# run
	interpreter.invoke()

	# get outpu tensor
	boxes = interpreter.get_tensor(output_details[0]['index'])
	labels = interpreter.get_tensor(output_details[1]['index'])
	scores = interpreter.get_tensor(output_details[2]['index'])
	num = interpreter.get_tensor(output_details[3]['index'])

	for i in range(boxes.shape[1]):
		if scores[0, i] > 0.5:
			box = boxes[0, i, :]
			x0 = int(box[1] * img_org.shape[1])
			y0 = int(box[0] * img_org.shape[0])
			x1 = int(box[3] * img_org.shape[1])
			y1 = int(box[2] * img_org.shape[0])
			box = box.astype(np.int)
			cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
			cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
			cv2.putText(img_org,
				   str(int(labels[0, i])),
				   (x0, y0),
				   cv2.FONT_HERSHEY_SIMPLEX,
				   1,
				   (255, 255, 255),
				   2)

#	cv2.imwrite('output.jpg', img_org)
	cv2.imshow('image', img_org)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	detect_from_camera()
	detect_from_image()

