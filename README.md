简述：
其中的merged_data.csv文件是原数据文件，其中的AQI为PM2.5的，地理空间特征为静态的，这里对其每一个时刻都进行广播操作了，同时AQI值不为None的网格为站点网格，其余网格为周边3*3的邻域网格，已排序好
data_process/Data下存储的是经过data_process.py处理后的数据文件，其中Kri_and_Knn_data.csv是用于克里金插值和KNN插值的数据集
拉取项目：  
git clone https://github.com/zhongcyi/ChongQing_AQI.git  
cd ChongQing_AQI   
环境配置：
conda create -n tf_gpu_2_10 python=3.10 -y  
conda activate tf_gpu_2_10  
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0  
pip install numpy==1.26.4
pip install "tensorflow==2.10"  
pip install scikit-learn
pip install pykrige
运行流程：  
cd data_process  
python data_process.py  
cd ..\Models  
python [Models_name].py 

