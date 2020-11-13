# Making Paper Reviewing Robust to Bid Manipulation Attacks
## 0. Dependencies and Data Download
### 0.1 Dependencies
```ortools, scipy, pytorch, numpy, tqdm```
### 0.2 Download raw data from Google drive
1. ```mkdir ./data```
2. The simulated paper bidding dataset can be downloaded from [here](https://drive.google.com/drive/folders/1khI9kaPy_8F0GtAzwR-48Jc3rsQmBhfe?usp=sharing). The data directory should be organized in the following way.
```
./data/raw_data/
       papers_dictionary.json
       reviewers_dictionary.json
       tensor_data.pl
       readme.txt
```

## 1. Data Preprocessing

Use the following script to generate the features for every pair of paper and reviewer.
```
bash scripts/init_and_data_process_script.sh
```

## 2 Model Training and Assignment Evaluation
This is an example script to train the linear regression model, evaluate its precision per reviewer / paper, and evaluate the assignment given by the regressor scores without any detection process.
```
bash scripts/assignment_script.sh
```

## 3 Attack
This is an example script to simulate white-box attack with colluding group size 5 and evaluate the success rate of this attack at each bin.

Comments for the arguments in the script:
- ```L```: the number of additional colluding people. 
- ```subsample_max```: the parameter **U** introduced in the paper.
- ```K```: the parameter that we only consider top **K** reviewer for each paper when we generate the final assignment.
- ```cheat_mode```: the cheating mode (```white_box``` / ```black_box```).
```
bash scripts/attack_script.sh
```

## 4 Detection Evaluation

### 4.1 TPR
This is an example script to evaluate our detection method against to the white-box attacking with colluding group size 5. The evaluation would run for different detection group sizes (1, 2, 3, 4, 5) together.

Comments for the arguments in the script: 
- ```L_attack```: the attack examples' additional colluding group size. 
- Other arguments have the same meaning as the one in 3.
```
bash scripts/detect_tpr_script.sh
```

### 4.2 FPR
An example script to get the fpr when we set the colluding group size to be detected as 5. 

Comments for the arguments in the script: 
- ```L```: the additional colluding group size set for detection. 
- Other arguments have the same meaning as the one in 3 and 4.1.

```
bash scripts/detect_fpr_script.sh
```

# Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

# License
This project is CC-BY-NC 4.0 licensed, as found in the LICENSE file.
