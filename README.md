## This repository is the official implementation of EIEA (Explicit-Implicit Entity Alignment Method in Multi-modal Knowledge Graphs).

## Environment
The essential package and recommened version to run the code:
`pip install -r EIEA-env.txt`

## Dataset
The MMEA-data and ECS-results can be download from [GoogleDrive](https://drive.google.com/drive/folders/1wfErYdAV93yxPtPHqkGanbmb_Ztv-LRU?usp=drive_link).
The original MMEA dataset can be download from MMKB and MMEA. 
Those files should be organized into the following file hierarchy:

- EIEA
  - data
    - ECS_results
      - seed0.2
        - FB15K-DB15K
        - FB15K-YAGO15K
      - seed0.5
      - seed0.8
    - MMEA_name
    - MMEA-data
      - seed0.2
        - FB15K-DB15K
        - FB15K-YAGO15K
      - seed0.5
      - seed0.8
  - src
    - DBP15K_src
    - ECS_compute

## Run EIEA
`sh mmkg.sh`
