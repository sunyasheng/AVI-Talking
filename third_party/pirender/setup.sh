cd Deep3DFaceRecon_pytorch/

~/azcopy copy "https://ussclowpriv100data.blob.core.windows.net/v-yashengsun/BFM.tar?sv=2021-10-04&st=2023-02-02T13%3A21%3A08Z&se=2024-05-03T13%3A21%3A00Z&sr=c&sp=racwdxltf&sig=vV5fXzUKmYaOdfapdzpk82lkU5eKwMYuP14qjD4akGs%3D" ./
mkdir -p ./checkpoints/model_name/
cd ./checkpoints/model_name/
~/azcopy copy "https://ussclowpriv100data.blob.core.windows.net/v-yashengsun/epoch_20.pth?sv=2021-10-04&st=2023-02-02T13%3A21%3A08Z&se=2024-05-03T13%3A21%3A00Z&sr=c&sp=racwdxltf&sig=vV5fXzUKmYaOdfapdzpk82lkU5eKwMYuP14qjD4akGs%3D" ./

tar xvf BFM.tar

cd nvdiffrast/
pip install .

cd ../..
pip install -r requirements.txt