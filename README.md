pyfreenect2
===========

Python bindings to [libfreenect2](https://github.com/OpenKinect/libfreenect2).

Requirements
---------

- Python2 (python3 support : https://github.com/LovelyHorse/py3freenect2)
- Numpy
- Scipy (as appropriated by python version) : 
- Python Imaging Library (used for scipy.misc.im* functions) : http://www.pythonware.com/products/pil/
- OpenCV

Installation
---------
`cd ~\path\to\pyfreenect2`

You need libfreenect2 installed, and also download [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) and put it in the `/examples` dir.

Add `export LD_LIBRARY_PATH=$HOME/freenect2/lib:$LD_LIBRARY_PATH` to your `~/.bashrc` file.

`pip install scipy opencv-python`, use `sudo` if necessary.

Try `sudo ln -s $HOME/freenect2/lib/libfreenect2.so.0.2.0 /usr/lib/libfreenect2.so` or check out [this issue](https://github.com/remexre/pyfreenect2/issues/11)

`sudo python setup.py install`.

Usage
---------

For usage, see `test.py`.


TODO List
---------
 * ~~Make `test.py` actually display the frames~~
 * Implement Registration.apply (in py_Registration_apply)
 * Make the pipeline argument of Freenect2Device's constructor actually do something
 * Test everything

You can probably find more TODOs in [Issues](https://github.com/tikiking1/pyfreenect2/issues) or by `grep -R TODO .`.
