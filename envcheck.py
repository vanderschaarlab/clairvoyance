def print_title(txt):
    print("=" * 80)
    print(">>> {}".format(txt))
    print("=" * 80)

print_title("Python info:")
import sys
print(sys.version)

print_title("Importing TensorFlow")
import tensorflow as tf

print_title("Importing PyTorch")
import torch

print_title("TensorFlow version:")
print(tf.__version__)  # pylint: disable=no-member

print_title("Checking TF CUDA (if releveant):")
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print_title("PyTorch version:")
print(torch.__version__)  # pylint: disable=no-member

print_title("Checking PyTorch CUDA:")
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("torch.backends.cudnn.version():", torch.backends.cudnn.version())
