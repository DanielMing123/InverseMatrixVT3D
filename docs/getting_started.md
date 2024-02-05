# NuScenes

## NuScenes 3D Occupancy Prediction Benchmark
**a. Train InverseMatrixVT3D with 8 GPUs for $256\times256\times32$ Resolution.**
```shell
bash tools/dist_train.sh configs/InverseMatrixVT3D_256_256_32.py 8
```
**b. Train InverseMatrixVT3D with 8 GPUs for $200\times200\times16$ Resolution.**
```shell
bash tools/dist_train.sh configs/InverseMatrixVT3D_200_200_16.py 8
```
**c. Test InverseMatrixVT3D with 8 GPUs for $256\times256\times32$ Resolution.**
```shell
bash tools/dist_test.sh configs/InverseMatrixVT3D_256_256_32.py work_dirs/InverseMatrixVT3D_256/epoch_24.pth 8
```
**d. Test InverseMatrixVT3D with 8 GPUs for $200\times200\times16$ Resolution.**
```shell
bash tools/dist_test.sh configs/InverseMatrixVT3D_200_200_16.py work_dirs/InverseMatrixVT3D_200/epoch_24.pth 8
```
