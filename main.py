import tensorflow as tf

print(tf.__version__)
print("GPU:", tf.test.gpu_device_name())
print("CUDA Version:", tf.test.is_built_with_cuda())
print("cuDNN Version:", tf.test.is_built_with_cudnn())

