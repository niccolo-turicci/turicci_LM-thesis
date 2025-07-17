# AlphaPullDown and AlphaBridge: a pipeline for PPI prediction and validation
This is the repository for my Master's degree thesis in Molecular Biology and Genetics at the Università di Pavia (Dipartimento di Biologia e Biotecnologie, L. Spallanzani). 

## Aim of the thesis
The goal is to create a protocol for efficient PPI prediction and validation by integrating two tools: AlphaPullDown (APD) and AlphaBridge (AB). 

## Workflow
![Workflow scheme](images/workflow.png)
1- produce PPI predictions (via APD)

2- modify the APD output to make it suited for AB

3- run AB to get the plots

## Command usage
### Converting APD output to ABridge
```python
python -m src --input ./your/input/folder --output ./your/output/folder
