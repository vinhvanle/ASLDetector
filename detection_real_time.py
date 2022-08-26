import cv2
import numpy as np
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

CUSTOM_MODEL_NAME = 'my_ssd_mobnet_tuned'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-56')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def print_detection(detections):
    for i in range(detections['num_detections']):
        if detections['detection_scores'][i] < 0.01:
            if detections['detection_classes'][i] == 1:
                print('A')
            elif detections['detection_classes'][i] == 2:
                print('B')
            elif detections['detection_classes'][i] == 3:
                print('C')
            elif detections['detection_classes'][i] == 4:
                print('D')
            elif detections['detection_classes'][i] == 5:
                print('E')
            elif detections['detection_classes'][i] == 6:
                print('F')
            elif detections['detection_classes'][i] == 7:
                print('G')
            elif detections['detection_classes'][i] == 8:
                print('H')
            elif detections['detection_classes'][i] == 9:
                print('I')
            elif detections['detection_classes'][i] == 10:
                print('J')
            elif detections['detection_classes'][i] == 11:
                print('K')
            elif detections['detection_classes'][i] == 12:
                print('L')
            elif detections['detection_classes'][i] == 13:
                print('M')
            elif detections['detection_classes'][i] == 14:
                print('N')
            elif detections['detection_classes'][i] == 15:
                print('O')
            elif detections['detection_classes'][i] == 16:
                print('P')
            elif detections['detection_classes'][i] == 17:
                print('Q')
            elif detections['detection_classes'][i] == 18:
                print('R')
            elif detections['detection_classes'][i] == 19:
                print('S')
            elif detections['detection_classes'][i] == 20:
                print('T')
            elif detections['detection_classes'][i] == 21:
                print('U')
            elif detections['detection_classes'][i] == 22:
                print('V')
            elif detections['detection_classes'][i] == 23:
                print('W')
            elif detections['detection_classes'][i] == 24:
                print('X')
            elif detections['detection_classes'][i] == 25:
                print('Y')
            elif detections['detection_classes'][i] == 26:
                print('Z')
            elif detections['detection_classes'][i] == 27:
                print('Airplane')
            elif detections['detection_classes'][i] == 28:
                print('Bus')
            elif detections['detection_classes'][i] == 29:
                print('CalmDown')
            elif detections['detection_classes'][i] == 30:
                print('Fine')
            elif detections['detection_classes'][i] == 31:
                print('Hello')
            elif detections['detection_classes'][i] == 32:
                print('Help')
            elif detections['detection_classes'][i] == 33:
                print('Home')
            elif detections['detection_classes'][i] == 34:
                print('IHateYou')
            elif detections['detection_classes'][i] == 35:
                print('ILoveYou')
            elif detections['detection_classes'][i] == 36:
                print('No')
            elif detections['detection_classes'][i] == 37:
                print('Okay')
            elif detections['detection_classes'][i] == 38:
                print('Pray')
            elif detections['detection_classes'][i] == 39:
                print('Ship')
            elif detections['detection_classes'][i] == 40:
                print('Time')
            elif detections['detection_classes'][i] == 41:
                print('Where')
            elif detections['detection_classes'][i] == 42:
                print('Yes')

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while cap.isOpened():
    ret, frame = cap.read()
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    classes = detections['detection_classes']
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']
    for i in range(min(5, boxes.shape[0])):
        if scores is None or scores[i] >0.9:
            if classes[i]+label_id_offset in category_index.keys():
                class_name = category_index[classes[i]+label_id_offset]['name']
                print(class_name)

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.9,
        agnostic_mode=False)



    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))





    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

