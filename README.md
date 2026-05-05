# TrustworthyML_2026
## Membership Inference Attack (MIA)
### Reproducing best result
##### 1. Clone Repository
##### git clone https://github.com/Meenakshi04081999/TrustworthyML_2026.git
##### cd TrustworthyML_2026

##### 2. Install Dependencies
##### pip install torch torchvision pandas numpy

##### 3. Download dataset and model
##### Download the following files and place them in the project root directory.
##### pub.pt, priv.pt and model.pth
##### (These are already provided in the resources).

##### 4. Set Paths
##### Open task_template.py and update API key, Pub_path, Priv_Path, Model_path, Output_CSV.

##### 5. Run the attack
##### python task_template.py
##### This will load the pretrained model, build the member feature bank, compute MIA score on priv ds, save results as submission.csv and finally submit to the leaderboard.
