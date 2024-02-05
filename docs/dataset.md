# NuScenes
Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). 

Generate annotation pickle files by running the following command:
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```
Using the above code will generate `nuscenes_infos_{train,val}.pkl`.

## NuScenes Occupancy Benchmark 
Download the SurroundOcc's 3D Occupancy annotations [HERE](https://github.com/weiyithu/SurroundOcc/blob/main/docs/data.md)
- only support training and validation set

**dataset structure**
```
InverseMatrixVT3D
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── occ_samples/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_train.pkl
|   |   ├── nuscenes_infos_val.pkl
|   |   ├── nuscenes_infos_test.pkl
```
