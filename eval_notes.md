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