#multi-gpu training
CUDA_VISIBLE_DEVICES=1 python train.py baseline_effnetb7_3planes_train12_eval3 & CUDA_VISIBLE_DEVICES=2 python train.py baseline_effnetb7_3planes_train13_eval2 & CUDA_VISIBLE_DEVICES=3 python train.py baseline_effnetb7_3planes_train32_eval1


