#wget -c https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
#tar xzf 1.10.0.tar.gz
#export CUB_HOME=$PWD/cub-1.10.0
#
#cd ~/
#git clone https://github.com/facebookresearch/pytorch3d.git
#cd pytorch3d
#pip install -e .

#export CUDA_HOME="/usr/local/cuda-10.2/"
#export LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH"
#export PATH="/usr/local/cuda-10.2/bin:$PATH"

## it requires nvcc version and cu102/torch-1.11.0%2Bcu102-cp38-cp38-linux_x86_64.whl torch version consistent
#pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'


#~/azcopy copy "https://ussclowpriv100data.blob.core.windows.net/v-yashengsun/ckpt.zip?sv=2021-10-04&st=2023-02-02T13%3A21%3A08Z&se=2024-05-03T13%3A21%3A00Z&sr=c&sp=racwdxltf&sig=vV5fXzUKmYaOdfapdzpk82lkU5eKwMYuP14qjD4akGs%3D" ./
#unzip ckpt.zip
#mv ckpt checkpoints

cd ~/
mkdir -p motion-latent-diffusion
cd motion-latent-diffusion
git init
git remote add origin 
git pull
