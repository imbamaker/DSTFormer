# DSTFormer
## 🚀 Quick Start
The project is developed under the following environment:
- Python 3.8.10
- PyTorch 2.0.0
- CUDA 12.2

For installation of the project dependencies, please run:
```pip install -r requirements.txt```

## Dataset

## Human3.6M

Preprocessing

1.Download the fine-tuned Stacked Hourglass detections of MotionBERT's preprocessed H3.6M data [here](https://onedrive.live.com/?id=A5438CD242871DF0!206&resid=A5438CD242871DF0!206&e=vobkjZ&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdkFkaDBMU2pFT2xnVTdCdVVaY3lhZnU4a3pjP2U9dm9ia2pa&cid=a5438cd242871df0) and unzip it to ```data/motion3d```.
2.Slice the motion clips by running the following python code in ```data/preprocess``` directory:

For 27 frames:
``` python h36m.py  --n-frames 27 ``` 

For 81 frames:
``` python h36m.py  --n-frames 81 ```

For 243 frames:
``` python h36m.py  --n-frames 243 ```


## MPI-INF-3DHP
 
Preprocessing

Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (data_train_3dhp.npz and data_test_3dhp.npz) should be located at ```data/motion3d directory```.



## Training

You can train Human3.6M with the following command:

```python train.py -config configs/h36m/DSTFormer-large.yaml```
```python train.py -config configs/h36m/DSTFormer-small.yaml```
```python train.py -config configs/h36m/DSTFormer-xsmall.yaml```

You can train MPI-INF-3DHP with the following command:

``` python train_3dhp.py --config configs/mpi/DSTFormer-large.yaml  ```
``` python train_3dhp.py --config configs/mpi/DSTFormer-large.yaml  ```
``` python train_3dhp.py --config configs/mpi/DSTFormer-large.yaml  ```


## Evaluation

For example if want to evalutae T = 243 model , we can run:

``` python train.py --eval-only --checkpoint checkpoint --checkpoint-file best_epoch.pth.tr --config configs/h36m/DSTFormer-large.yaml ```




