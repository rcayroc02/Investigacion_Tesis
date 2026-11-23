# TMGF full model
# market
python train.py --conf configs/TMGF_full.yml TASK_NAME TMGF_training_market DATASET.ROOT_DIR ./datasets DATASET.NAME Market1501 MODEL.SIE_CAMERA 6 MEMORY_BANK.POS_K 3
# duke
python train.py --conf configs/TMGF_full.yml TASK_NAME TMGF_training_duke DATASET.ROOT_DIR ./datasets DATASET.NAME DukeMTMC-reID MODEL.SIE_CAMERA 8 MEMORY_BANK.POS_K 2
# msmt
python train.py --conf configs/TMGF_full.yml TASK_NAME TMGF_training_msmt DATASET.ROOT_DIR ./datasets DATASET.NAME MSMT17 MODEL.SIE_CAMERA 15 MEMORY_BANK.POS_K 3



#occluded duke CONFIG 1
python train.py --conf configs/TMGF_full.yml TASK_NAME TMGF_training_occd_duke DATASET.ROOT_DIR ./datasets DATASET.NAME OCCDuke MODEL.SIE_CAMERA 8 MEMORY_BANK.POS_K 2
#occluded duke CONFIG 2
python train.py --conf configs/exp_cfg2_re_intensified.yml TASK_NAME cfg2_re_intensified_occd_duke DATASET.ROOT_DIR ./datasets DATASET.NAME OCCDuke MODEL.SIE_CAMERA 8 MEMORY_BANK.POS_K 2
#occluded duke CONFIG 3
python train.py --conf configs/exp_cfg3_stripe_directed.yml TASK_NAME cfg3_stripe_directed_occd_duke DATASET.ROOT_DIR ./datasets DATASET.NAME OCCDuke MODEL.SIE_CAMERA 8 MEMORY_BANK.POS_K 2