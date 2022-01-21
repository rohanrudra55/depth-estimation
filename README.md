# Depth Estimation

Depth estimation from stereo images using disparity calculation algorithms.

## Python Version

### Requirements

- Python version 3.8
- Install python modules via requirements.txt

```bash
pip install -r requirements.txt
```

### Usage

For features check help

```bash
python3 src/Python/Depth_Estimation.py --h
```

---

## C++ Version

### Requirements

- [Cmake](https://cmake.org/download/) 2.8 or higher
- OpenCV
- PCL 1.2

### Usage

```bash
mkdir build
cd build
cmake ..
make
./stereo
./stereo filename
.stereo filename