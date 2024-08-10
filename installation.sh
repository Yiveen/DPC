#pytorch version is not strict to this
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
conda install cmake==3.22.3
pip install pytorch-lightning==1.5.10 #tested

pip install opencv-python --prefer-binary # also pip install opencv-python, but failed on my cluster

conda install -c plotly psutil requests python-kaleido --yes
pip install cython==0.29.20 autowrap ninja tables ply ilock
# delete some useless packages
pip install h5py pydocstyle plotly psutil xvfbwrapper yapf mypy plyfile neuralnet-pytorch imageio pyinstrument pairing robust_laplacian pymesh trimesh cmake "ray[tune]" "pytorch-lightning-bolts>=0.2.5" pyrr gdist neptune-client neptune-contrib iopath autowrap torchsummary py-goicp
conda install "notebook>=5.3" "ipywidgets>=7.2" flake8 black flake8 -y
conda install pytorch-metric-learning -c metric-learning -c pytorch -y
pip install addict
pip install open3d-python open3d potpourri3d
pip install point-cloud-utils
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

#  need to sepecify the cuda version
export PATH=/shared-network/yliu/cuda_11.3/bin:$PATH
export CPATH=/shared-network/yliu/cuda_11.3/include:$CPATH
export LD_LIBRARY_PATH=/shared-network/yliu/cuda_11.3/lib64:$LD_LIBRARY_PATH
export CUDA=cu113
export TORCH=1.11.0

pip install torch-scatter==2.0.9 torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
pip install chardet==5.1.0