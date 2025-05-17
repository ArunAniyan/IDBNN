```

April 5th 2025
![image](https://github.com/user-attachments/assets/1c9ddbdc-5ee7-4a6d-9ec2-8a894528ce29)

![image](https://github.com/user-attachments/assets/f1798522-9aa9-46c9-b4e0-38ad4748bd33)
Fresh training after all modifications on April 5th - ook about 1 hr and 15 mins.

March 31 2025 version Train the CDBNN, predict with it and train the adbnn and finally predcit with it.
Later you can skip the training process and use the predcit modes on your own images.

git pull && python cdbnn.py --data_name galaxies --input_path  galaxies.zip --mode train
git pull && python cdbnn.py --data_name galaxies --input_path  galaxies.zip --mode predict
&& python adbnn.py --file_path data/galaxies/galaxies.csv --mode train
&& python adbnn.py --file_path data/galaxies/galaxies.csv --mode predict
Already up to date.
The code has some bug in directly handling torchvision files. So recommendation is to use Get_Torchvision_images function instead
Updated on March 30/2025 Stable version
2025-03-31 21:05:54,190 - INFO - Logging setup complete. Log file: logs/prediction_20250331_210554.log
2025-03-31 21:05:54,834 - INFO - Loaded unified checkpoint from data/galaxies/checkpoints/galaxies_unified.pth
2025-03-31 21:05:54,834 - INFO - Layer sizes: [32, 64, 128, 128, 128, 128]
2025-03-31 21:05:55,055 - WARNING - Using default clustering temperature 1.0
2025-03-31 21:05:55,057 - INFO - Model loaded successfully with clustering parameters.
2025-03-31 21:05:55,058 - INFO - Dataset created with 12 images
2025-03-31 21:05:55,058 - INFO - Using default output path: data/galaxies/galaxies.csv
2025-03-31 21:05:55,058 - INFO - Starting prediction process...
2025-03-31 21:05:57,429 - INFO - Processing 12632 images with batch size 128
Predicting features: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 99/99 [00:18<00:00,  5.47it/s]
2025-03-31 21:06:15,539 - INFO - Predictions saved to data/galaxies/galaxies.csv
2025-03-31 21:06:15,541 - INFO - Predictions saved to data/galaxies/galaxies.csv
DBNN Dataset Processor
========================================
Processing galaxies in predict mode
The model components are loaded from Model/Best_Histogram_galaxies_components.pkl
Attempting to load weights file Model/Best_Histogram_galaxies_weights.json
Weights file Model/Best_Histogram_galaxies_weights.json loaded succesfully
Loaded best weights from Model/Best_Histogram_galaxies_weights.json
Using device: cuda
Attempting to load weights file Model/Best_Histogram_galaxies_weights.json
Weights file Model/Best_Histogram_galaxies_weights.json loaded succesfully
Loaded 0 training samples from best model
Loaded best weights from Model/Best_Histogram_galaxies_weights.json
[DEBUG] Generating feature combinations after filtering:
- n_features: 128
- group_size: 2
- max_combinations: 100000
- bin_sizes: [128]
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loading cached feature combinations from training_data/galaxies/feature_combinations.pkl
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loaded feature combinations: torch.Size([8128, 2])
adbnn.py:2012: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  feature_indices = torch.unique(torch.cat([torch.tensor(pair, device=self.device)
Starting prediction...
Processing galaxies in predict mode
Loaded label encoder with classes: ['NonRings' 'Rings']
Loaded likelihood_params from components file
[DEBUG] Weight initialization complete. Structure:
- Number of classes: 2
- Class 0: 8128 feature pairs
- Class 1: 8128 feature pairs
Using model for dataset: galaxies
Loaded label encoder with classes: ['NonRings' 'Rings']
Loaded likelihood_params from components file
[DEBUG] Weight initialization complete. Structure:
- Number of classes: 2
- Class 0: 8128 feature pairs
- Class 1: 8128 feature pairs
Prediction batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 99/99 [06:18<00:00,  3.83s/it]
Confusion Matrix and Class-wise Accuracy for [Prediction Results]:
Actual/PredictedNonRingsRings   Accuracy
-----------------------------------------
NonRings       10015   1422     87.57%
Rings          218     977      81.76%
-----------------------------------------
Overall Accuracy: 87.02%


#----------------------------------------Previous ---------------------
Single-line command for training
>> git pull && python cdbnn.py --data_name galaxies --input_path galaxies.zip --mode train &&  python adbnn.py --file_path data/galaxies/galaxies.csv --mode train
OR
>> git pull && python cdbnn.py && python adbnn.py

Enter mode (train/predict) [train]:

Enter dataset type (torchvision/custom) [custom]:
Enter dataset name/path [galaxies.zip]:
Enable inverse DBNN mode? (y/n) [y]:
Enter encoder type (cnn/autoenc) [cnn]:
Enter batch size [128]:
Enter number of epochs [20]:
Enter output directory [data]:
2025-03-17 01:43:32,564 - INFO - Extracting compressed file: galaxies.zip
Create train/test split? (y/n): n
2025-03-17 01:43:42,423 - INFO - Dataset processed: train_dir=data/galaxies/train, test_dir=None
2025-03-17 01:43:42,423 - INFO - Generating/verifying configurations...
2025-03-17 01:43:42,423 - INFO - Starting configuration generation for dataset: galaxies
2025-03-17 01:43:42,423 - INFO - Generating main configuration...
2025-03-17 01:43:42,424 - INFO - Using image properties from config file
2025-03-17 01:43:42,487 - INFO - Found existing main config, merging...
2025-03-17 01:43:42,488 - INFO - Main configuration saved: data/galaxies/galaxies.json
2025-03-17 01:43:42,488 - INFO - Generating dataset configuration...
2025-03-17 01:43:42,488 - INFO - Found existing dataset config, merging...
2025-03-17 01:43:42,489 - INFO - Dataset configuration saved: data/galaxies/galaxies.conf
2025-03-17 01:43:42,489 - INFO - Generating DBNN configuration...
2025-03-17 01:43:42,489 - INFO - Found existing DBNN config, merging...
2025-03-17 01:43:42,489 - INFO - DBNN configuration saved: data/galaxies/adaptive_dbnn.conf

Configuring General Enhancement Parameters:
Enable KL divergence clustering? (y/n) [y]:
Enter KL divergence weight (0-1) [0.1]:
Enable class encoding? (y/n) [y]:
Enter classification weight (0-1) [0.1]:
Enter clustering temperature (0.1-2.0) [1.0]:
Enter minimum cluster confidence (0-1) [0.7]:
Enable phase 2 training (clustering and fine-tuning)? (y/n) [y]:

Configuring Enhancement Features for General Mode:
You can enable any combination of features

Enable Astronomical features (star detection, galaxy structure preservation)? (y/n) [n]: y
Astronomical features added.
Enable Medical features (tissue boundary, lesion detection)? (y/n) [n]:
Enable Agricultural features (texture analysis, damage detection)? (y/n) [n]:

Current Enhancement Configuration:

Astronomical Features:
- Components: structure_preservation, detail_preservation, star_detection, galaxy_features, kl_divergence
- Weights: detail_weight: 1.0, structure_weight: 0.8, edge_weight: 0.7

Learning Rates:
- Phase 1: 0.001
- Phase 2: 0.0005
2025-03-17 01:44:57,245 - INFO - Initializing general enhanced model...
2025-03-17 01:44:57,246 - INFO - Creating model with enhancements: astronomical, medical, agricultural
2025-03-17 01:44:57,246 - INFO - Layer sizes: [32, 64, 128, 256, 256, 256]

Ready to start training. Proceed? (y/n): y

2025-03-17 02:09:21,540 - INFO - Starting model training...
2025-03-17 02:09:21,541 - INFO - Starting/Resuming Phase 1: Pure reconstruction training
2025-03-17 02:09:21,543 - INFO - Initialized new unified checkpoint
Phase 1 - Epoch 1: 100%|█████████████████████████████████████████| 99/99 [00:24<00:00,  3.96it/s, loss=0.7696, best=inf]
2025-03-17 02:09:46,751 - INFO - Saved state phase1 to unified checkpoint
2025-03-17 02:09:46,752 - INFO - New best model for phase1 with loss: 0.7696
2025-03-17 02:09:46,752 - INFO - Phase 1 - Epoch 1: Loss = 0.7696, Best = 0.7696
Phase 1 - Epoch 2: 100%|█████████████████████████████████████████| 99/99 [00:25<00:00,  3.94it/s, loss=0.6646, best=0.7696]
2025-03-17 02:10:12,640 - INFO - Saved state phase1 to unified checkpoint
2025-03-17 02:10:12,640 - INFO - New best model for phase1 with loss: 0.6646
2025-03-17 02:10:12,640 - INFO - Phase 1 - Epoch 2: Loss = 0.6646, Best = 0.6646
Phase 1 - Epoch 3: 100%|██████████████████████████████████████████| 99/99 [00:24<00:00,  4.06it/s, loss=0.6643, best=0.6646]
2025-03-17 02:10:37,445 - INFO - Saved state phase1 to unified checkpoint
2025-03-17 02:10:37,446 - INFO - New best model for phase1 with loss: 0.6643
2025-03-17 02:10:37,446 - INFO - Phase 1 - Epoch 3: Loss = 0.6643, Best = 0.6643
Phase 1 - Epoch 4: 100%|█████████████████████████████████████████| 99/99 [00:23<00:00,  4.22it/s, loss=0.6637, best=0.6643]
2025-03-17 02:11:01,334 - INFO - Saved state phase1 to unified checkpoint
2025-03-17 02:11:01,335 - INFO - New best model for phase1 with loss: 0.6637
2025-03-17 02:11:01,335 - INFO - Phase 1 - Epoch 4: Loss = 0.6637, Best = 0.6637
....

2025-03-17 03:50:10,818 - INFO - Features saved to data/galaxies/galaxies.csv
2025-03-17 03:50:10,818 - INFO - Total features saved: 135
2025-03-17 03:50:10,819 - INFO - Processing completed successfully!
DBNN Dataset Processor
========================================
usage: adbnn.py [-h] [--file_path [FILE_PATH]] [--mode {train,train_predict,invertDBNN}]

Process ML datasets

optional arguments:
  -h, --help            show this help message and exit
  --file_path [FILE_PATH]
                        Path to dataset file or folder
  --mode {train,train_predict,invertDBNN}
                        Mode to run the network: train, train_predict, or invertDBNN.
usage: adbnn.py [-h] [--file_path [FILE_PATH]] [--mode {train,train_predict,invertDBNN}]

Process ML datasets

optional arguments:
  -h, --help            show this help message and exit
  --file_path [FILE_PATH]
                        Path to dataset file or folder
  --mode {train,train_predict,invertDBNN}
                        Mode to run the network: train, train_predict, or invertDBNN.
============================================================
Dataset: galaxies
Config file: data/galaxies/galaxies.conf
Data file: data/galaxies/galaxies.csv
============================================================
Dataset Information:
Dataset name: galaxies
Configuration file: data/galaxies/galaxies.conf (4.4 KB)
Data file: data/galaxies/galaxies.csv (18691.4 KB)
Model type: Histogram
Target column: target
Number of columns: 129
Excluded features: 0
Features:
  feature_0
  feature_1
  feature_2
  feature_3
  feature_4
  ... and 124 more

Process this dataset? (y/n): y
Processing dataset: galaxies
Loaded best weights from Model/Best_Histogram_galaxies_weights.json
Using device: cuda
Using original data order (no shuffling required)
Loaded best weights from Model/Best_Histogram_galaxies_weights.json
Using original data order (no shuffling required)
[DEBUG] Generating feature combinations after filtering out features with high cardinality set by the conf file:
- n_features: 128
- group_size: 2
- max_combinations: 10000
- bin_sizes: [21]
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loading cached feature combinations from data/galaxies/training_data/galaxies/feature_combinations.pkl
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loaded feature combinations: torch.Size([1830, 2])
Dataset shape: torch.Size([12632, 128])
Bin sizes: [64]
Using original data order (no shuffling required)
Adaptive training started at: 2025-03-17 03:50:34
target
 Initial data shape: X=(12632, 129), y=12632
Number of classes in data = [0 1]
<bound method NDFrame.head of        feature_0  feature_1  feature_2  feature_3  feature_4  feature_5  ...  feature_124  feature_125  feature_126  feature_127  target  original_index
0       0.038956  -0.116702  -0.036961   0.119280  -0.085445   0.049149  ...     0.280519    -0.031874    -0.003397     0.052938       0               0
1       0.638382  -0.032868  -0.038245   0.089761  -0.202669   0.525085  ...     0.558681    -0.018332    -0.024747     0.557497       0               1
2       0.492175  -0.170648  -0.038416   0.085148  -0.200476   0.378307  ...     0.414804    -0.012266    -0.063049     0.523771       0               2
3       0.730011  -0.149996  -0.039406   0.090917  -0.247599   0.605390  ...     0.709854    -0.015759    -0.021829     0.796731       0               3
4       0.487268  -0.081182  -0.038580   0.098442  -0.149054   0.397495  ...     0.571912    -0.031604    -0.004842     0.546710       0               4
...          ...        ...        ...        ...        ...        ...  ...          ...          ...          ...          ...     ...             ...
12627   0.313439  -0.308851  -0.030555   0.091439  -0.263172   0.248972  ...     0.584872    -0.044785     0.008434     0.318932       1           12627
12628   0.303588  -0.102883  -0.038293   0.107569  -0.013486   0.253513  ...     0.544572    -0.012600    -0.002367     0.439162       0           12628
12629   0.415428  -0.255094  -0.034059   0.084471  -0.032345   0.361214  ...     0.624342    -0.039793     0.023227     0.443271       1           12629
12630   0.192983  -0.214469  -0.041186   0.091015   0.337450   0.118773  ...     0.242265    -0.057006    -0.039371     0.326471       0           12630
12631   0.501341  -0.088689  -0.039442   0.091163  -0.034817   0.405114  ...     0.632384    -0.028419    -0.009838     0.541648       0           12631

[12632 rows x 130 columns]>
[DEBUG] Generating feature combinations after filtering out features with high cardinality set by the conf file:
- n_features: 110
- group_size: 2
- max_combinations: 10000
- bin_sizes: [21]
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loading cached feature combinations from data/galaxies/training_data/galaxies/feature_combinations.pkl
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loaded feature combinations: torch.Size([1830, 2])
Dataset shape: torch.Size([12632, 110])
Bin sizes: [64]
Loading previous model state
No previous model found - starting fresh
Initializing fresh model
[DEBUG] Generating feature combinations after filtering out features with high cardinality set by the conf file:
- n_features: 110
- group_size: 2
- max_combinations: 10000
- bin_sizes: [21]
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loading cached feature combinations from data/galaxies/training_data/galaxies/feature_combinations.pkl
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loaded feature combinations: torch.Size([1830, 2])
Computing pairwise likelihoods...
Processing feature pairs: 100%|██████████████████████████████████████████████████| 1830/1830 [00:01<00:00, 956.46it/s]
[DEBUG] Weight initialization complete. Structure:
- Number of classes: 2
- Class 0: 1830 feature pairs
- Class 1: 1830 feature pairs
Initializing new training set with minimum samples
Round 1/20
Training set size: 4
Test set size: 12628
Training epochs:   0%|                                                                                                                                     | 0/1000 [00:00<?, ?it/s Prediction batches: 100%|█████████████████████████████████| 1/1 [00:00<00:00,  2.60it/s]
Training epochs:   0%|                                                           | 1/1000 [00:00<13:24,  1.24it/s, train_err=0.0000 (best: 0.0000), train_acc=1.0000 (best: 1.0000)  Early stopping.
Training epochs:   0%|                                                           | 1/1000 [00:00<13:24,  1.24it/s, train_err=0.0000 (best: 0.0000), train_acc=1.0000 (best: 1.0000)]
Prediction batches: 100%|████████████████████████████████████████████| 395/395 [02:34<00:00,  2.55it/s]
---------------------------------------------------------------------------------------
The best combined accuracy has improved from 0.0 to 0.9426852438252058
---------------------------------------------------------------------------------------

Saved model components to Model/BestHistogram_galaxies_components.pkl
Confusion Matrix and Class-wise Accuracy for [Training Data]:
Actual/Predicted0       1       Accuracy
-----------------------------------------
0              2       0       100.00%
1              0       2       100.00%
-----------------------------------------
Overall Accuracy: 100.00%
Best Overall Accuracy till now is: 94.27%
Confusion Matrix and Class-wise Accuracy for [Test Data]:
Actual/Predicted0       1       Accuracy
-----------------------------------------
0              11055   380      96.68%
1              344     849      71.17%
-----------------------------------------
Overall Accuracy: 94.27%
Best Overall Accuracy till now is: 94.27%
Confusion Matrix and Class-wise Accuracy for [Combined Data]:
Actual/Predicted0       1       Accuracy
-----------------------------------------
0              11057   380      96.68%
1              344     851      71.21%
-----------------------------------------
Overall Accuracy: 94.27%
Best Overall Accuracy till now is: 94.27%
Training accuracy: 1.0000
Last testing data is saved to galaxies_Last_testing.csv
Last training data is saved to galaxies_Last_training.csv
Saved model and data due to improved training accuracy
Total samples selected: 403
Resetting to initial conditions of best round None
Added 403 new samples to training set
Round 2/20
Training set size: 407
Test set size: 12225
Training epochs:   0%|                                                                                                                                     | 0/1000 [00:00<?, ?it/s Prediction batches: 100%|███████████████████████████████████████████████████| 13/13 [00:05<00:00,  2.53it/s]
Training epochs:   0%|                                                         | 1/1000 [00:10<2:49:29, 10.18s/it, train_err=0.9902 (best: 0.9902),
train_acc=0.0098 (best: 0.0098) Prediction batches: 100%|████████████████████████████████████████████████████| 13/13 [00:04<00:00,  2.60it/s]
Training epochs:   0%|                                                        | 2/1000 [01:15<11:53:22, 42.89s/it, train_err=0.3317 (best: 0.3317),
 train_acc=0.6683 (best: 0.6683) Prediction batches: 100%|█████████████████████████████████████████████████████| 13/13 [00:04<00:00,  2.60it/s]
Training epochs:   0%|▏                                                       | 3/1000 [01:45<10:11:59, 36.83s/it, train_err=0.2187 (best: 0.2187),
 train_acc=0.7813 (best: 0.7813) Prediction batches: 100%|██████████████████████████████████████████████████████| 13/13 [00:05<00:00,  2.52it/s]
Training epochs:   0%|▏                                                        | 4/1000 [02:08<8:43:12, 31.52s/it, train_err=0.1941 (best: 0.1941),
 train_acc=0.8059 (best: 0.8059) Prediction batches: 100%|████████████████████████████████████████████████████████| 13/13 [00:05<00:00,  2.51it/s]
Training epochs:   0%|▎                                                        | 5/1000 [02:30<7:45:45, 28.09s/it, train_err=0.1794 (best: 0.1794),
train_acc=0.8206 (best: 0.8206)  Early stopping.
Training epochs:   0%|▎                                                        | 5/1000 [02:30<8:20:41, 30.19s/it, train_err=0.1794 (best: 0.1794),
train_acc=0.8206 (best: 0.8206)]
Prediction batches: 100%|████████████████████████████████████████████████████████████████| 395/395 [02:34<00:00,  2.55it/s]
---------------------------------------------------------------------------------------
The best combined accuracy has improved from 0.9426852438252058 to 0.9690468651044966
---------------------------------------------------------------------------------------
aved model components to Model/BestHistogram_galaxies_components.pkl
Confusion Matrix and Class-wise Accuracy for [Training Data]:
Actual/Predicted0       1       Accuracy
-----------------------------------------
0              169     23       88.02%
1              50      165      76.74%
-----------------------------------------
Overall Accuracy: 82.06%
Best Overall Accuracy till now is: 96.90%
Confusion Matrix and Class-wise Accuracy for [Test Data]:
Actual/Predicted0       1       Accuracy
-----------------------------------------
0              11022   223      98.02%
1              95      885      90.31%
-----------------------------------------
Overall Accuracy: 97.40%
Best Overall Accuracy till now is: 96.90%
Confusion Matrix and Class-wise Accuracy for [Combined Data]:
Actual/Predicted0       1       Accuracy
-----------------------------------------
0              11191   246      97.85%
1              145     1050     87.87%
-----------------------------------------
Overall Accuracy: 96.90%
Best Overall Accuracy till now is: 96.90%
Training accuracy: 0.8206
Improved training accuracy to 0.8206
Last testing data is saved to galaxies_Last_testing.csv
Last training data is saved to galaxies_Last_training.csv
Saved model and data due to improved training accuracy
Total samples selected: 49
Resetting to initial conditions of best round None
Added 49 new samples to training set
Round 3/20
Training set size: 456
Test set size: 12176
Training epochs:   0%|  | 0/1000 [00:00<?, ?it/s Prediction batches: 100%|█████████████████████████████████████████████████████| 15/15 [00:05<00:00,  2.51it/s]
Training epochs:   0%|                                                         | 1/1000 [00:11<3:17:13, 11.85s/it, train_err=0.2675 (best: 0.2675),
train_acc=0.7325 (best: 0.7325) Prediction batches: 100%|████████████████████████████████████████████| 15/15 [00:06<00:00,  2.50it/s]
Training epochs:   0%|                                                         | 2/1000 [00:41<6:11:26, 22.33s/it, train_err=0.2566 (best: 0.2566),
train_acc=0.7434 (best: 0.7434) Prediction batches: 100%|████████████████████████████████████████████| 15/15 [00:05<00:00,  2.51it/s]
Training epochs:   0%|▏                                                        | 3/1000 [01:10<7:01:19, 25.36s/it, train_err=0.2500 (best: 0.2500),
train_acc=0.7500 (best: 0.7500) Prediction batches: 100%|████████████████████████████████████████████| 15/15 [00:05<00:00,  2.52it/s]
Training epochs:   0%|▏                                                        | 4/1000 [01:38<7:21:35, 26.60s/it, train_err=0.2390 (best: 0.2390),
train_acc=0.7610 (best: 0.7610) Prediction batches: 100%|████████████████████████████████████████████| 15/15 [00:05<00:00,  2.58it/s]
Training epochs:   0%|▎                                                        | 5/1000 [02:06<7:27:26, 26.98s/it, train_err=0.2390 (best: 0.2390),
train_acc=0.7610 (best: 0.7610)  Early stopping.
Training epochs:   0%|▎                                                        | 5/1000 [02:06<7:00:01, 25.33s/it, train_err=0.2390 (best: 0.2390),
train_acc=0.7610 (best: 0.7610)]
Prediction batches: 100%|███████████████████████████████████████████████████████████████████████| 395/395 [02:35<00:00,  2.54it/s]
---------------------------------------------------------------------------------------
The best combined accuracy has improved from 0.9690468651044966 to 0.9701551614946169
---------------------------------------------------------------------------------------
Saved model components to Model/BestHistogram_galaxies_components.pkl
Saved model components to Model/BestHistogram_galaxies_components.pkl
Confusion Matrix and Class-wise Accuracy for [Training Data]:
Actual/Predicted0       1       Accuracy
-----------------------------------------
0              176     34       83.81%
1              75      171      69.51%
-----------------------------------------
Overall Accuracy: 76.10%
Best Overall Accuracy till now is: 97.02%
Confusion Matrix and Class-wise Accuracy for [Test Data]:
Actual/Predicted0       1       Accuracy
-----------------------------------------
0              11022   205      98.17%
1              63      886      93.36%
-----------------------------------------
Overall Accuracy: 97.80%
Best Overall Accuracy till now is: 97.02%
Confusion Matrix and Class-wise Accuracy for [Combined Data]:
Actual/Predicted0       1       Accuracy
-----------------------------------------
0              11198   239      97.91%
1              138     1057     88.45%
-----------------------------------------
Overall Accuracy: 97.02%
Best Overall Accuracy till now is: 97.02%
Training accuracy: 0.7610
Improved training accuracy to 0.7610
Last testing data is saved to galaxies_Last_testing.csv
Last training data is saved to galaxies_Last_training.csv

----------------Now there is no improvement, and the code runs till persistence is exhausted.
Finally...

Confusion Matrix and Class-wise Accuracy for [Training Data]:
Actual/Predicted0       1       Accuracy
-----------------------------------------
0              176     104      62.86%
1              138     171      55.34%
-----------------------------------------
Overall Accuracy: 58.91%
Best Overall Accuracy till now is: 97.02%
Confusion Matrix and Class-wise Accuracy for [Test Data]:
Actual/Predicted0       1       Accuracy
-----------------------------------------
0              11021   136      98.78%
1              0       886     100.00%
-----------------------------------------
Overall Accuracy: 98.87%
Best Overall Accuracy till now is: 97.02%
Confusion Matrix and Class-wise Accuracy for [Combined Data]:
Actual/Predicted0       1       Accuracy
-----------------------------------------
0              11197   240      97.90%
1              138     1057     88.45%
-----------------------------------------
Overall Accuracy: 97.01%
Best Overall Accuracy till now is: 97.02%
Training accuracy: 0.5891
No significant overall improvement. Adaptive patience: 5/5
No improvement in accuracy after 5 rounds of adding samples.
Best training accuracy achieved: 0.9702
Stopping adaptive training.
Adaptive training started at: 2025-03-17 04:43:20
Adaptive training ended at: 2025-03-17 05:46:11
Total adaptive training time: 3770.87 seconds
[DEBUG] Generating feature combinations after filtering out features with high cardinality set by the conf file:
- n_features: 110
- group_size: 2
- max_combinations: 10000
- bin_sizes: [21]
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loading cached feature combinations from data/galaxies/training_data/galaxies/feature_combinations.pkl
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loaded feature combinations: torch.Size([1830, 2])
Dataset shape: torch.Size([12632, 110])
Bin sizes: [64]
Prediction batches: 100%|████████████████████████████████████████████████████████████████| 25/25 [00:09<00:00,  2.58it/s]
Processing complete!
Time taken: 3937.5 seconds
Results saved to: data/galaxies/training_data/galaxies/galaxies_predictions.csv
Training log saved to: data/galaxies/training_data/galaxies/galaxies_log.csv
Processed 12632 samples with 129 features
Excluded 19 features
Loaded best weights from Model/Best_Histogram_galaxies_weights.json
Using device: cuda
Using original data order (no shuffling required)
Loaded best weights from Model/Best_Histogram_galaxies_weights.json
Using original data order (no shuffling required)
[DEBUG] Generating feature combinations after filtering out features with high cardinality set by the conf file:
- n_features: 110
- group_size: 2
- max_combinations: 10000
- bin_sizes: [21]
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loading cached feature combinations from data/galaxies/training_data/galaxies/feature_combinations.pkl
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loaded feature combinations: torch.Size([1830, 2])


The output of the training using the model on 27/03/2025

---------------------------------------------------------------------------------------
The best combined accuracy has improved from 0.9975459151361621 to 0.9994458518049398
---------------------------------------------------------------------------------------
Saved model components to Model/BestHistogram_galaxies_components.pkl
Confusion Matrix and Class-wise Accuracy for [Training Data]:
Actual/PredictedNonRingsRings   Accuracy
-----------------------------------------
NonRings       2       0       100.00%
Rings          1       383      99.74%
-----------------------------------------
Overall Accuracy: 99.74%
Best Overall Accuracy till now is: 99.94%
Confusion Matrix and Class-wise Accuracy for [Test Data]:
Actual/PredictedNonRingsRings   Accuracy
-----------------------------------------
NonRings       11429   6        99.95%
Rings          0       811     100.00%
-----------------------------------------
Overall Accuracy: 99.95%
Best Overall Accuracy till now is: 99.94%
Confusion Matrix and Class-wise Accuracy for [Combined Data]:
Actual/PredictedNonRingsRings   Accuracy
-----------------------------------------
NonRings       11431   6        99.95%
Rings          1       1194     99.92%
-----------------------------------------
Overall Accuracy: 99.94%
Best Overall Accuracy till now is: 99.94%
Training accuracy: 0.9974
Improved training accuracy to 0.9974
Last testing data is saved to galaxies_Last_testing.csv
Last training data is saved to galaxies_Last_training.csv
Saved model and data due to improved training accuracy
 Batch size updated to- Optimal batch size: 4096
Total samples selected: 0
No suitable new samples found. Training complete.
Adaptive training started at: 2025-03-27 18:14:00
Adaptive training ended at: 2025-03-27 20:03:20
Total adaptive training time: 6560.59 seconds
[DEBUG] Generating feature combinations after filtering out features with high cardinality set by the conf file:

Dataset shape: torch.Size([12632, 110])
Bin sizes: [64]


```
