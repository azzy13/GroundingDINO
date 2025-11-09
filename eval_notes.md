swinb best 
0016        209 49.5% 0.222 69.1% 87.1% 57.3% 87.9% 57.9%  11  6  4  106 228 1206        2863

===== 0018 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML  FM  FP  FN num_objects
0018        322 76.4% 0.154 88.2% 84.7% 91.9% 85.4% 92.6%   5 12  0  14 214 100        1354

===== 0019 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML   FM   FP   FN num_objects
0019       1059 46.5% 0.255 66.2% 74.5% 59.6% 79.4% 63.5%  41 20  9  204 1154 2559        7015

===== 0020 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML  FM  FP   FN num_objects
0020        837 61.9% 0.156 75.9% 81.0% 71.5% 85.2% 75.1%  12 37 10  43 718 1367        5497

====== AVERAGE ======
      num_frames      mota      motp      idf1  ...  num_fragmentations  num_false_positives  num_misses  num_objects
0015       376.0  0.334949  0.204488  0.643691  ...                45.0                650.0       441.0       1651.0
0016       209.0  0.495285  0.222401  0.691238  ...               106.0                228.0      1206.0       2863.0
0018       322.0  0.764402  0.154179  0.881644  ...                14.0                214.0       100.0       1354.0
0019      1059.0  0.464861  0.254835  0.661861  ...               204.0               1154.0      2559.0       7015.0
0020       837.0  0.618519  0.155854  0.759401  ...                43.0                718.0      1367.0       5497.0
AVG        560.6  0.535603  0.198351  0.727567  ...                82.4                592.8      1134.6       3676.0

[6 rows x 15 columns]
OPTUNA:MOTA=0.535603 IDF1=0.727567

swinb OG

===== 0015 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML  FM  FP  FN num_objects
0015        376 53.2% 0.241 71.4% 73.9% 69.0% 78.8% 73.6%   9  7  3  45 327 436        1651

===== 0016 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML  FM  FP  FN num_objects
0016        209 54.5% 0.287 70.9% 82.1% 62.4% 86.2% 65.6%  16  6  2  92 300 986        2863

===== 0018 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML  FM  FP  FN num_objects
0018        325 77.2% 0.156 85.8% 85.3% 86.3% 88.5% 89.5%   9  6  0  11 158 142        1354

===== 0019 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML   FM   FP   FN num_objects
0019       1059 41.6% 0.287 61.5% 63.8% 59.3% 72.8% 67.6%  52 20  6  226 1774 2271        7015

===== 0020 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML  FM  FP   FN num_objects
0020        837 46.8% 0.191 62.1% 84.8% 48.9% 90.7% 52.4%  13  9 34  40 294 2618        5497

====== AVERAGE ======
      num_frames      mota      motp      idf1  ...  num_fragmentations  num_false_positives  num_misses  num_objects
0015       376.0  0.532405  0.241231  0.714062  ...                45.0                327.0       436.0       1651.0
0016       209.0  0.545232  0.287235  0.709127  ...                92.0                300.0       986.0       2863.0
0018       325.0  0.771787  0.155697  0.858297  ...                11.0                158.0       142.0       1354.0
0019      1059.0  0.415966  0.286874  0.614941  ...               226.0               1774.0      2271.0       7015.0
0020       837.0  0.467892  0.190529  0.620531  ...                40.0                294.0      2618.0       5497.0
AVG        561.2  0.546656  0.232313  0.703391  ...                82.8                570.6      1290.6       3676.0

[6 rows x 15 columns]
OPTUNA:MOTA=0.546656 IDF1=0.703391

swinb epoch 50


===== 0015 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML  FM  FP  FN num_objects
0015        376 33.4% 0.201 65.7% 62.0% 69.9% 65.0% 73.3%   7  6  4  43 651 441        1651

===== 0016 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML   FM  FP   FN num_objects
0016        209 49.5% 0.223 69.1% 87.0% 57.3% 87.8% 57.8%  10  6  4  106 230 1207        2863

===== 0018 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML  FM  FP FN num_objects
0018        322 77.0% 0.155 88.4% 85.1% 92.0% 85.8% 92.8%   5 14  0  14 208 98        1354

===== 0019 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML   FM   FP   FN num_objects
0019       1059 46.4% 0.255 66.2% 74.5% 59.6% 79.4% 63.5%  47 20  9  205 1159 2557        7015

===== 0020 =====
     num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML  FM  FP   FN num_objects
0020        837 62.0% 0.156 76.0% 81.0% 71.5% 85.2% 75.3%  12 37 10  47 718 1357        5497

====== AVERAGE ======
      num_frames      mota      motp      idf1  ...  num_fragmentations  num_false_positives  num_misses  num_objects
0015       376.0  0.334343  0.201248  0.657175  ...                43.0                651.0       441.0       1651.0
0016       209.0  0.494586  0.222865  0.690672  ...               106.0                230.0      1207.0       2863.0
0018       322.0  0.770310  0.154527  0.884315  ...                14.0                208.0        98.0       1354.0
0019      1059.0  0.463578  0.255168  0.662445  ...               205.0               1159.0      2557.0       7015.0
0020       837.0  0.620338  0.156127  0.759633  ...                47.0                718.0      1357.0       5497.0
AVG        560.6  0.536631  0.197987  0.730848  ...                83.0                593.2      1132.0       3676.0

[6 rows x 15 columns]
OPTUNA:MOTA=0.536631 IDF1=0.730848

## MOT17
====== AVERAGE ======
               num_frames      mota     motp     idf1  ...  num_fragmentations  num_false_positives    num_misses   num_objects
MOT17-02-DPM   600.000000  0.000000      NaN  0.00000  ...            0.000000             0.000000   9393.000000   9393.000000
MOT17-04-DPM  1050.000000  0.000000      NaN  0.00000  ...            0.000000             0.000000  39821.000000  39821.000000
MOT17-05-DPM   837.000000  0.494651  0.22225  0.62279  ...           25.000000           502.000000   1892.000000   4767.000000
MOT17-09-DPM   525.000000  0.000000      NaN  0.00000  ...            0.000000             0.000000   3745.000000   3745.000000
MOT17-10-DPM   654.000000  0.000000      NaN  0.00000  ...            0.000000             0.000000  10560.000000  10560.000000
MOT17-11-DPM   900.000000  0.000000      NaN  0.00000  ...            0.000000             0.000000   7518.000000   7518.000000
MOT17-13-DPM   750.000000  0.000000      NaN  0.00000  ...            0.000000             0.000000  10024.000000  10024.000000
AVG            759.428571  0.070664  0.22225  0.08897  ...            3.571429            71.714286  11850.428571  12261.142857

[8 rows x 15 columns]

============================================================
📈 Summary: MOTA=7.07% | IDF1=8.90%
============================================================

OPTUNA:MOTA=0.070664 IDF1=0.088970

#  --box_threshold 0.25 --text_threshold 0.10 --track_thresh 0.30

====== AVERAGE ======
               num_frames      mota      motp      idf1  ...  num_fragmentations  num_false_positives    num_misses   num_objects
MOT17-02-DPM   600.000000  0.000000       NaN  0.000000  ...                 0.0             0.000000   9393.000000   9393.000000
MOT17-04-DPM  1050.000000  0.000000       NaN  0.000000  ...                 0.0             0.000000  39821.000000  39821.000000
MOT17-05-DPM   837.000000  0.604993  0.230755  0.706483  ...                49.0           946.000000    886.000000   4767.000000
MOT17-09-DPM   525.000000  0.000000       NaN  0.000000  ...                 0.0             0.000000   3745.000000   3745.000000
MOT17-10-DPM   654.000000  0.000000       NaN  0.000000  ...                 0.0             0.000000  10560.000000  10560.000000
MOT17-11-DPM   900.000000  0.000000       NaN  0.000000  ...                 0.0             0.000000   7518.000000   7518.000000
MOT17-13-DPM   750.000000  0.000000       NaN  0.000000  ...                 0.0             0.000000  10024.000000  10024.000000
AVG            759.428571  0.086428  0.230755  0.100926  ...                 7.0           135.142857  11706.714286  12261.142857

[8 rows x 15 columns]

============================================================
📈 Summary: MOTA=8.64% | IDF1=10.09%
============================================================

OPTUNA:MOTA=0.086428 IDF1=0.100926

## AP numbers
# how 2 run
python3 eval_visdrone.py --config_file config/GroundingDINO_SwinB_cfg.py --checkpoint /isis/home/hasana3/vlmtest/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth --coco_anno /isis/home/hasana3/vlmtest/GroundingDINO/dataset/visdrone/VisDrone2019-MOT-val/dataset_coco.json --image_root /isis/home/hasana3/vlmtest/GroundingDINO/dataset/visdrone/VisDrone2019-MOT-val

# soft finetune - 10 epoch 1e-4
DONE (t=2.67s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.710
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.029
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.706

 # aggro fine tune 2e-4
 Test: Total time: 0:06:41 (0.1412 s / it)
Averaged stats: 
Accumulating evaluation results...
DONE (t=2.62s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.702
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.620
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.027
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.195
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.695

 # original swinb groundingdino checkpoint
 Accumulating evaluation results...
DONE (t=2.77s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.304
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.595
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.078
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.305
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.192
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.419
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.251
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.661

Results saved to ./output/results.json

# MOTA on aggro ft checkpoint
===== uav0000086_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000086_00000_v        464 65.5% 0.244 74.8% 81.2% 69.3% 88.5% 75.5%  33 40 13 27  399 2196 5497       22410       16880

===== uav0000117_02622_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000117_02622_v        349 39.5% 0.235 49.7% 65.9% 39.9% 83.4% 50.5% 150 25 82 27  266 1528 7529       15224        7545

===== uav0000137_00458_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000137_00458_v        233 47.8% 0.220 60.7% 68.5% 54.6% 80.8% 64.3% 272 56 39 65  458 3226 7535       21118       13311

===== uav0000182_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000182_00000_v        363 10.9% 0.247 54.0% 53.6% 54.3% 55.6% 56.2%  28 30 30 26  128 4270 4157        9494        5309

===== uav0000305_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP   FN num_objects num_matches
uav0000305_00000_v        184 52.8% 0.154 73.8% 80.6% 68.2% 81.5% 68.9%  19 22 16 13  41 759 1505        4839        3315

===== uav0000339_00001_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM  FP   FN num_objects num_matches
uav0000339_00001_v        275 43.7% 0.226 55.0% 74.2% 43.7% 87.6% 51.6%  60 26 19 20  165 748 4934       10203        5209
====== AVERAGE ======
                    num_frames      mota      motp      idf1  ...  num_false_positives   num_misses   num_objects   num_matches
uav0000086_00000_v  464.000000  0.655243  0.244074  0.747899  ...          2196.000000  5497.000000  22410.000000  16880.000000
uav0000117_02622_v  349.000000  0.395231  0.234655  0.496912  ...          1528.000000  7529.000000  15224.000000   7545.000000
uav0000137_00458_v  233.000000  0.477555  0.219680  0.607483  ...          3226.000000  7535.000000  21118.000000  13311.000000
uav0000182_00000_v  363.000000  0.109438  0.247091  0.539553  ...          4270.000000  4157.000000   9494.000000   5309.000000
uav0000305_00000_v  184.000000  0.528208  0.154079  0.738468  ...           759.000000  1505.000000   4839.000000   3315.000000
uav0000339_00001_v  275.000000  0.437224  0.226360  0.550185  ...           748.000000  4934.000000  10203.000000   5209.000000
AVG                 311.333333  0.433817  0.220990  0.613417  ...          2121.166667  5192.833333  13881.333333   8594.833333

[7 rows x 17 columns]

============================================================
📈 Summary: MOTA=43.38% | IDF1=61.34%
============================================================

OPTUNA:MOTA=0.433817 IDF1=0.613417

✅ Complete! Results saved to: outputs/visdrone_val_2025-11-04_0611

# MOTA on best ft checkpoint
===== uav0000086_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000086_00000_v        464 59.5% 0.240 74.1% 86.5% 64.8% 89.8% 67.2%  21 33 21 26  420 1709 7340       22410       15049

===== uav0000117_02622_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM  FP   FN num_objects num_matches
uav0000117_02622_v        349 34.6% 0.224 45.8% 72.0% 33.6% 87.8% 41.0% 106 18 92 24  199 867 8985       15224        6133

===== uav0000137_00458_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000137_00458_v        233 47.8% 0.212 60.0% 73.6% 50.6% 85.5% 58.8% 229 50 44 66  436 2112 8692       21118       12197

===== uav0000182_00000_v =====
                   num_frames MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM   FP   FN num_objects num_matches
uav0000182_00000_v        363 9.6% 0.230 45.3% 54.1% 39.0% 56.7% 40.8%  10 17 41 28  69 2957 5619        9494        3865

===== uav0000305_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP   FN num_objects num_matches
uav0000305_00000_v        184 55.6% 0.166 74.9% 84.1% 67.4% 84.9% 68.0%  14 23 16 12  32 587 1547        4839        3278

===== uav0000339_00001_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM  FP   FN num_objects num_matches
uav0000339_00001_v        275 38.4% 0.206 50.0% 81.9% 36.0% 94.0% 41.3%  28 21 28 16  107 270 5991       10203        4184

====== AVERAGE ======
                    num_frames      mota      motp      idf1  ...  num_false_positives   num_misses   num_objects  num_matches
uav0000086_00000_v  464.000000  0.595270  0.239684  0.741024  ...               1709.0  7340.000000  22410.000000      15049.0
uav0000117_02622_v  349.000000  0.345901  0.224216  0.458039  ...                867.0  8985.000000  15224.000000       6133.0
uav0000137_00458_v  233.000000  0.477555  0.212031  0.599899  ...               2112.0  8692.000000  21118.000000      12197.0
uav0000182_00000_v  363.000000  0.095639  0.230224  0.453142  ...               2957.0  5619.000000   9494.000000       3865.0
uav0000305_00000_v  184.000000  0.556107  0.166252  0.748566  ...                587.0  1547.000000   4839.000000       3278.0
uav0000339_00001_v  275.000000  0.383613  0.206180  0.500102  ...                270.0  5991.000000  10203.000000       4184.0
AVG                 311.333333  0.409014  0.213098  0.583462  ...               1417.0  6362.333333  13881.333333       7451.0

[7 rows x 17 columns]

============================================================
📈 Summary: MOTA=40.90% | IDF1=58.35%
============================================================

OPTUNA:MOTA=0.409014 IDF1=0.583462

✅ Complete! Results saved to: outputs/visdrone_val_2025-11-04_0622

# Aggro ft checkpoint all sequences
📂 Found 7 GT files and 7 result files

===== uav0000086_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000086_00000_v        464 65.5% 0.244 74.8% 81.2% 69.4% 88.4% 75.5%  32 40 13 27  403 2220 5490       22410       16888

===== uav0000117_02622_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000117_02622_v        349 39.5% 0.235 49.4% 65.4% 39.7% 83.4% 50.6% 156 25 81 28  267 1529 7522       15224        7546

===== uav0000137_00458_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000137_00458_v        233 47.7% 0.220 60.4% 68.0% 54.3% 80.8% 64.5% 309 56 38 66  479 3238 7505       21118       13304

===== uav0000182_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM   FP   FN num_objects num_matches
uav0000182_00000_v        363 10.6% 0.247 53.8% 53.4% 54.2% 55.4% 56.2%  28 30 30 26  128 4299 4157        9494        5309

===== uav0000268_05773_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP   FN num_objects num_matches
uav0000268_05773_v        978 22.0% 0.129 37.7% 93.5% 23.6% 93.5% 23.6%   1  5 38  7  18 214 9981       13068        3086

===== uav0000305_00000_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT  FM  FP   FN num_objects num_matches
uav0000305_00000_v        184 52.7% 0.154 73.8% 80.5% 68.2% 81.3% 68.9%  19 22 16 13  41 765 1505        4839        3315

===== uav0000339_00001_v =====
                   num_frames  MOTA  MOTP  IDF1   IDP   IDR  Prcn  Rcll IDs MT ML PT   FM  FP   FN num_objects num_matches
uav0000339_00001_v        275 43.8% 0.226 55.1% 74.3% 43.8% 87.7% 51.7%  67 26 19 20  165 742 4930       10203        5206

====== AVERAGE ======
                    num_frames      mota      motp      idf1  ...  num_false_positives  num_misses   num_objects   num_matches
uav0000086_00000_v  464.000000  0.654529  0.244114  0.748303  ...          2220.000000      5490.0  22410.000000  16888.000000
uav0000117_02622_v  349.000000  0.395231  0.234929  0.493723  ...          1529.000000      7522.0  15224.000000   7546.000000
uav0000137_00458_v  233.000000  0.476655  0.219823  0.603650  ...          3238.000000      7505.0  21118.000000  13304.000000
uav0000182_00000_v  363.000000  0.106383  0.247132  0.537689  ...          4299.000000      4157.0   9494.000000   5309.000000
uav0000268_05773_v  978.000000  0.219773  0.129350  0.377054  ...           214.000000      9981.0  13068.000000   3086.000000
uav0000305_00000_v  184.000000  0.526968  0.154061  0.737973  ...           765.000000      1505.0   4839.000000   3315.000000
uav0000339_00001_v  275.000000  0.437518  0.226401  0.551239  ...           742.000000      4930.0  10203.000000   5206.000000
AVG                 406.571429  0.402437  0.207973  0.578519  ...          1858.142857      5870.0  13765.142857   7807.714286

[8 rows x 17 columns]

============================================================
📈 Summary: MOTA=40.24% | IDF1=57.85%
============================================================

OPTUNA:MOTA=0.402437 IDF1=0.578519

✅ Complete! Results saved to: outputs/visdrone_val_2025-11-04_0657

# lower lr finetuning checkpoint all sequeneces
