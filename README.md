# AlphaPullDown and AlphaBridge: a pipeline for PPI prediction and validation
This is the repository with the code for my Master's degree thesis in Molecular Biology and Genetics at the Università di Pavia (Dipartimento di Biologia e Biotecnologie, L. Spallanzani). 

## Aim of the thesis
The goal is to create a protocol for efficient PPI prediction and validation by integrating two tools: AlphaPullDown (APD) and AlphaBridge (AB). It includes benchmarking of the two separate tools (AlphaPullDown and AlphaBridge) and creating a script that makes the intergration between the two seamless. 

## Ideal workflow
![Workflow scheme](images/workflow.png)
1- produce PPI predictions (via APD)

2- modify the APD output to make it suited for AB

3- run AB to get the plots

## 2. APD output conversion (APD -> AB)
There are several scripts, specific for every new file that has to be created.
Each script is commented and explained internally. 
  ### full_data.py
  This creates the full_data.json file starting from the .pdb (ranked_X.pdb) file, the PAE matrix adn the .pkl file; this done for each of the five predicted structures.
  ### summary_confidences.py
  This needs the follwing files to create the summary_confidences.json: .pdb structure, .pkl file, confidence_model.json file, PAE matrix and the ranking_debug.json. 
  ### job_request.py
  Only needs the .pdb structure and some manual information, plus the job name, specified when launchig the command with the specific argument. This creates the only job_request.json file needed. 
  ### pdb_to_cif.py
  This converts the .pdb structures into .cif (AF3-like) format; the only one accepted by AlphaBridge. 
  ### ranking.py
  This script renames each summary_confidences.json , and all the realtive files accordingly, based on the ipTM score. In this way ABridge uses the more confident structure to operate. 

## Command usage
### Converting APD output to ABridge
```python
python -m src --input ./your/input/folder --output ./your/output/folder --jobname nameofjob
