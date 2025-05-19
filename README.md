拉取项目：
git clone https://github.com/PKU-VCL-3DV/SLAM3R.git
cd SLAM3R
环境配置：
 conda create -n tf_gpu_2_10 python=3.10 -y  
 conda activate tf_gpu_2_10  
 conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0  
 pip install numpy==1.26.4
 python -m pip install "tensorflow==2.10"

