# AlphaPullDown and AlphaBridge: a workflow for PPI prediction and validation
This is the repository with the code for my Master's degree thesis in Molecular Biology and Genetics at the Università di Pavia (Dipartimento di Biologia e Biotecnologie, L. Spallanzani). 

## Aim of the thesis
The goal is to create a protocol for efficient PPI prediction and validation by integrating two tools: AlphaPullDown (APD) and AlphaBridge (AB). It includes benchmarking of the two separate tools (AlphaPullDown and AlphaBridge) and creating a script that makes the intergration between the two seamless. 

# Ideal workflow
![Workflow scheme](images/workflow.png)
1- produce PPI predictions (via APD)

2- modify the APD output to make it suited for AB

3- run AB to get the plots

## 0. Install tools
Guides, scripts and installation requirements can be found at the following pages:
 APD --> https://github.com/KosinskiLab/AlphaPulldown 
 AB --> https://github.com/PDB-REDO/AlphaBridge 

## 1. AlphaPullDown (APD)
Need to run on **HPC** with a **batch job**; so it needs **job request scripts**. 
These script are in the `turicci_LM-thesis/alphapulldown/src` folder. There's one for running APD in _custom_ mode and one for _pulldown_ mode.  
## 2. APD output conversion (APD -> AB)
There are several scripts, specific for every new file that has to be created.
Each script is commented and explained internally. 
```python
python -m src --input ./your/input/folder --output ./your/output/folder --jobname nameofjob
```
_--input_ the folder with the APD output

_--output_ where to save the code output (converted folder). It will create output_folder and temp_folder inside the specified one. 

_--jobname_ nem of the job to assure reproducibility, trackability and consistency.

  ### full_data.py
  This creates the full_data.json file starting from the .pdb (ranked_X.pdb) file, the PAE matrix adn the .pkl file; this done for each of the five predicted structures.
  ### summary_confidences.py
  This needs the follwing files to create the summary_confidences.json: .pdb structure, .pkl file, confidence_model.json file, PAE matrix and the ranking_debug.json. 
  ### job_request.py
  Only needs the .pdb structure and some manual information, plus the job name, specified when launchig the command with the specific argument. This creates the only job_request.json file needed. 
  ### pdb_to_cif.py
  This converts the .pdb structures into .cif (AF3-like) format; the only one accepted by AlphaBridge. 
  ### ranking.py
  This script renames each summary_confidences.json , and all the realtive files accordingly, based on the ipTM score. In this way ABridge uses the structure with the most confident interface to operate. 

## 3. AlphaBridge (AB)
Python scripts are in the `turicci_LM-thesis/alphabridge` folder. These scripts can be ran in HPC with an interactive job; or even locally. 
First, to detect interacting residues:
```python
python define_interfaces.py -i <PATH/TO/AF3_FOLDER>
```
Then, to create the ribbon plots (circos):
```python
python alphabridge_circos.py <PATH/TO/AF3_FOLDER/AlphaBridge>
```

## BONUS TIP 💡:
To get the same AF3 coloring of pLDDT in PyMol:
```bash
set_color n0, [0.051, 0.341, 0.827]
set_color n1, [0.416, 0.796, 0.945]
set_color n2, [0.996, 0.851, 0.212]
set_color n3, [0.992, 0.490, 0.302]
color n0, b < 100; color n1, b < 90
color n2, b < 70;  color n3, b < 50
```

You're welcome 😜
