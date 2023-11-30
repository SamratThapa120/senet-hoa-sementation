# apt-get update -y
# apt-get install tmux -y
pip install -U openmim
mim install mmengine
pip uninstall opencv-python
pip install opencv-python==4.8.0.74
apt update && apt install -y libsm6 libxext6
apt-get install -y libxrender-dev
mim install mmcv==2.1.0
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .