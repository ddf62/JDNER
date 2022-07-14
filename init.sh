export CONDA_ENV_NAME=pyt
echo $CONDA_ENV_NAME
conda create -n $CONDA_ENV_NAME python=3.7
#在shell开启conda环境，以前写脚本没有这一句都不能执行conda命令
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME
#需要的环境：requirements.txt
pip install -r requirements.txt
