import sys
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

f = open('inspect_checkpoint.log', 'w')
sys.stdout = f
sys.stderr = f

ckpt = tf.train.get_checkpoint_state('../original_checkpoint')
chkp.print_tensors_in_checkpoint_file(ckpt.model_checkpoint_path, tensor_name='', all_tensors=True)

