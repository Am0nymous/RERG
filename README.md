# RERG
The code describes the article "RERG: Reinforced Evidence Reasoning with Graph
Neural Network for Table-based Fact Verification"

Step 1: Convert the tensorflow version TAPAS into pytorch

1.1 Download the pre-trained TAPAS model from https://storage.googleapis.com/tapas_models/2020_10_07/tapas_inter_masklm_large.zip

then put it into convert_model/Tapas_model

1.2 run the script convert_tf2_torch.py to get the pytorch version tapas model torch_tapas_model.bin

Altivate Step: fine-tune the tapas model on the MultiNLI corpus

the MultiNLI corpus can be obtained from https://cims.nyu.edu/~sbowman/multinli/, we remove the netural examples and take one epoch steps for fine-tuning.

Step 2: Tokenization and graph construction

run the script create_data.py

Step 3: training model on TABFACT dataset

run the script train.py

Step 4: evaluate test datasets

run the script evaluate.py







