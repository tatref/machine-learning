# Intro
Commands starting with **#** must be run as root

Commands starting with **$** must be run as simple user

Commands starting **(venv)** must be run from inside a virtualenv

# Setup for Debian 8.X Jessie

    $ cat /etc/debian_version 
    8.5
    $ uname -a
    Linux pc 3.16.0-4-amd64 #1 SMP Debian 3.16.7-ckt25-2+deb8u3 (2016-07-02) x86_64 GNU/Linux
    $ lspci  | grep -i geforce
    01:00.0 VGA compatible controller: NVIDIA Corporation GK104 [GeForce GTX 760] (rev a1)


# Install apt packages

    $ sudo apt-get update && apt-get dist-upgrade -y
    $ sudo apt-get install -y build-essential python3-dev python3-virtualenv \
        libopenblas-dev gfortran virtualenv pkg-config libfreetype6-dev \
        libpng12-dev git

# Create and activate the virtualenv

    $ virtualenv -p python3 venv
    $ . ./venv/bin/activate

# Install python packages

    (venv) pip install numpy scipy
    (venv) pip install scikit-learn
    (venv) pip install jupyter
    (venv) pip install pandas
    (venv) pip install matplotlib
    (venv) pip install theano
    (venv) pip install lasagne
    (venv) pip install grip

# Install Tensorflow /w CUDA 7.5
Dependencies

    apt-get install g++ libxi6 libxi-dev libglu1-mesa libglu1-mesa-dev libxmu6

Download `cuda_7.5.18_linux.run` from nvidia website

Disable nouveau if enabled

    $ cat /etc/modprobe.d/blacklist-nouveau.conf
    blacklist nouveau
    options nouveau modeset=0
    # update-initramfs -u
    
Reboot in runlevel 3

Install CUDA

    # chmod +x cuda_7.5.18_linux.run
    # ./cuda_7.5.18_linux.run
    # (next, next...)
    # reboot

Check driver

    # lsmod | grep nvi
    nvidia               8555203  43 
    drm                   249998  3 nvidia
    i2c_core               46012  3 drm,i2c_i801,nvidia

You can also install cuDNN (for optimized deep learning functions), see nvidia's website (https://developer.nvidia.com/rdp/cudnn-download). Download `cudnn-7.0-linux-x64-v4.0-prod.tgz`, since Tensorflow does not support cudnn 5.0 at the time of writing (Tensorfloow 0.9, august 2016)

    # tar xvzf cudnn-7.0-linux-x64-v4.0-prod.tgz
    # cp cuda/include/cudnn.h /usr/local/include/
    # cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/

Test Cuda

    $ export PATH=$PATH:/usr/local/cuda/bin
    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    $ cd NVIDIA_CUDA-7.5_Samples/0_Simple/vectorAdd
    $ make
    $ ./vectorAdd
    ...
    Test PASSED
    Done

To install Tensorflow with CUDA, see https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#pip-installation

Test Tensorflow with CUDA (from Tensorflow documentation)

    $ python
    ...
    >>> import tensorflow as tf
    I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.so locally
    I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.so locally
    I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.so locally
    I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcuda.so locally
    I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcurand.so locally
    >>> hello = tf.constant('Hello, TensorFlow!')
    >>> sess = tf.Session()
    >>> print(sess.run(hello))
    Hello, TensorFlow!
    >>> a = tf.constant(10)
    >>> b = tf.constant(32)
    >>> print(sess.run(a + b))
    42

You might want to add the following to your virtualenv activate script (if using CUDA)

    export PATH=$PATH:/usr/local/cuda/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Launch jupyter

    (venv) jupyter-notebook

# What next?
* jupyter examples
* numpy examples
