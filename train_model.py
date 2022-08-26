"""Import Dependencies"""
import os
import wget
import git
import object_detection
from zipfile import ZipFile
import shutil
import command
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

"""Set up paths"""

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
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

for path in paths.values():
    if not os.path.exists(path):
        os.makedirs(path)

"""Create label map"""
labels = [{'name': 'A', 'id': 1},
              {'name': 'B', 'id': 2},
              {'name': 'C', 'id': 3},
              {'name': 'D', 'id': 4},
              {'name': 'E', 'id': 5},
              {'name': 'F', 'id': 6},
              {'name': 'G', 'id': 7},
              {'name': 'H', 'id': 8},
              {'name': 'I', 'id': 9},
              {'name': 'J', 'id': 10},
              {'name': 'K', 'id': 11},
              {'name': 'L', 'id': 12},
              {'name': 'M', 'id': 13},
              {'name': 'N', 'id': 14},
              {'name': 'O', 'id': 15},
              {'name': 'P', 'id': 16},
              {'name': 'Q', 'id': 17},
              {'name': 'R', 'id': 18},
              {'name': 'S', 'id': 19},
              {'name': 'T', 'id': 20},
              {'name': 'U', 'id': 21},
              {'name': 'V', 'id': 22},
              {'name': 'W', 'id': 23},
              {'name': 'X', 'id': 24},
              {'name': 'Y', 'id': 25},
              {'name': 'Z', 'id': 26},
              {'name': 'Airplane', 'id': 27},
              {'name': 'Bus', 'id': 28},
              {'name': 'CalmDown', 'id': 29},
              {'name': 'Fine', 'id': 30},
              {'name': 'Hello', 'id': 31},
              {'name': 'Help', 'id': 32},
              {'name': 'Home', 'id': 33},
              {'name': 'IHateYou', 'id': 34},
              {'name': 'ILoveYou', 'id': 35},
              {'name': 'No', 'id': 36},
              {'name': 'Okay', 'id': 37},
              {'name': 'Pray', 'id': 38},
              {'name': 'Ship', 'id': 39},
              {'name': 'Time', 'id': 40},
              {'name': 'Where', 'id': 41},
              {'name': 'Yes', 'id': 42}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')


"""Generate TF records"""
#Generate train.record
train_command = 'python '+files['TF_RECORD_SCRIPT']+' -x '+os.path.join(paths['IMAGE_PATH'],'train')+' -l '+ files['LABELMAP'] +' -o '+os.path.join(paths['ANNOTATION_PATH'],'train.record')
#Generate test.record
test_command= 'python '+files['TF_RECORD_SCRIPT']+' -x '+os.path.join(paths['IMAGE_PATH'],'test')+' -l '+ files['LABELMAP'] +' -o '+os.path.join(paths['ANNOTATION_PATH'],'test.record')
os.system(train_command)
os.system(test_command)

"""Copy and configure pipeline.config"""
shutil.copy(os.path.join(paths['PRETRAINED_MODEL_PATH'],PRETRAINED_MODEL_NAME,'pipeline.config'),os.path.join(paths['CHECKPOINT_PATH']))
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 8
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
    f.write(config_text)


"""Train model"""


steps = input("How many steps")
freeze_graph = str(input("Do you want to freeze graph, Y/n"))
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
train_command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'],str(steps))

os.system(train_command)


if freeze_graph == "Y":
    FREEZE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py ')
    freeze_command = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(FREEZE_SCRIPT ,files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'], paths['OUTPUT_PATH'])
    os.system((freeze_command))
else:
    pass

