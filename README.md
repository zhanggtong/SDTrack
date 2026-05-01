# SDTrack  


## Install the environment
Use the Anaconda
```
conda create -n sdtrack python=3.8
conda activate sdtrack
bash install.sh
```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```
And you need to set the model paths for BERT in the configuration file.

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
     
        -- got10k
            |-- test
            |-- train
                  ...
            |-- val
        -- TNL2K
            |-- train
            |-- test
        
        -- trackingnet
             |-- TRAIN_0
             |-- TRAIN_1
             ...
             |-- TRAIN_11
             |-- TEST
        -- RefCOCO14
        -- OTB_Lang   
        
            
   ```
            
**Notice** ：The files lasot_train_concise, got-10k_train_concise, and tnl2k_train_concise come from [DTVLT](http://videocube.aitestunion.com/) and contain partial language descriptions from the three datasets.
## Training
Download pre-trained [HiViT-Base weights](https://drive.google.com/file/d/1VZQz4buhlepZ5akTcEvrA3a_nxsQZ8eQ/view?usp=share_link) and put it under `$PROJECT_ROOT$/pretrained_models` (see [HiViT](https://github.com/zhangxiaosong18/hivit) for more details).
Download pre-trained [AQAtrack weights](https://github.com/GXNU-ZhongLab/AQATrack) and put it under `$PROJECT_ROOT$/pretrained_models` .

```
bash train.sh
```


## Test
```
python test_epoch.py
```

## Evaluation 
```
python tracking/analysis_results.py
```


## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX2080Ti GPU.

```
# Profiling SDTrack-ep150-full-256
python tracking/profile_model.py --script aqatrack --config SDTrack-ep150-full-256
# Profiling SDTrack-ep150-full-384
python tracking/profile_model.py --script aqatrack --config SDTrack-ep150-full-384
```



