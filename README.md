This repository houses all code for reproducing the results of the paper:

```
CLIP: A Dataset for Extracting Action Items for Physicians from Hospital Discharge Notes
James Mullenbach, Yada Pruksachatkun, Sean Adler, Jennifer Seale, Jordan Swartz, Greg McKelvey, Hui Dai, Yi Yang and David Sontag
ACL 2021
```

# Setup

1. (Optional, recommended) Create and activate a virtual environment
2. `pip install -r requirements.txt`. 

# Data

The data will be hosted by PhysioNet, as it is based on MIMIC, which has access controls for privacy reasons and is also hosted by PhysioNet. Interested users will have to complete a short training course and sign a DUA with PhysioNet to gain access. 

The link to the dataset will go here when the associated submission is finished with copy editing. 

# Data processing

After downloading and unpacking the data, `cd data` and run `python convert_json_to_csv.py` to create sentence-level data that the training scripts can read.

[step to split into train/val/test]

# Experiments

All training scripts are in `sentclf/`. Bash scripts to invoke these scripts in the manner they were run to produce the paper results are found in `train_{modelname}.sh`. With the exception of the bag of words model, these scripts all will run training for multiple seeds and report final results on the test set, which can then be used to compute the table entries from the paper. 
