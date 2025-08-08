import os
import json
import pickle
import glob
import numpy as np
import math
import re
import itertools


# --- 4 - Creates summary_confidence.json file ---
def convert(obj):   # makes sure all data can be handled by the pipeline (reproducibility/flexibility)
    if isinstance(obj, np.ndarray):
        return [convert(i) for i in obj.tolist()]
    if isinstance(obj, (np.float32, np.float64, float)):
        if math.isnan(obj):
            return 0.0
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert(i) for i in obj]
    return obj

def round_floats(obj, ndigits=2):   # avoids having long values 
    if isinstance(obj, float):
        return round(obj, ndigits)
    elif isinstance(obj, list):
        return [round_floats(i, ndigits) for i in obj]
    elif isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    else:
        return obj

def get_chain_residue_ranges_from_pdb(pdb_file):   # looks for chain bundaries inside the .pdb file: lines starting with ATOM (stores first and last value)
    chain_residues = {}
    chain_order = []
    atom_line = re.compile(r"^ATOM\s+\d+\s+\S+\s+\S+\s+(\S)\s+(\d+)")
    idx = 0
    seen = set()
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21]
                resnum = int(line[22:26])
                key = (chain, resnum)
                if key not in seen:
                    seen.add(key)
                    if chain not in chain_residues:
                        chain_residues[chain] = [idx, idx]
                        chain_order.append(chain)
                    else:
                        chain_residues[chain][1] = idx
                    idx += 1
    for chain in chain_residues:
        start, end = chain_residues[chain]
        chain_residues[chain] = (start, end + 1)
    return chain_residues, chain_order

def compute_chain_pair_pae_min(pae, chain_ranges, chain_order):   # creates a matrix with the minimum pae value from each residue pair (lower pae = higher confidence)
    n = len(chain_order)
    pae_min = []
    for i in range(n):
        row = []
        start_i, end_i = chain_ranges[chain_order[i]]
        for j in range(n):
            start_j, end_j = chain_ranges[chain_order[j]]
            submatrix = pae[start_i:end_i, start_j:end_j]
            row.append(float(np.min(submatrix)))
        pae_min.append(row)
    return pae_min

def compute_chain_ptm_from_pae(pae_matrix, chain_ranges, chain_order):   # converts the average pae value for each chain into a ptm score (formula derived from AlphaFold)
    
    # here I calculate the values needed for the ptm_score formula (d0)
    L_target = pae_matrix.shape[0]
    d0 = 1.24 * np.cbrt(L_target - 15) - 1.8
    
    chain_ptm = []
    for chain in chain_order:
        start, end = chain_ranges[chain]
        submatrix = pae_matrix[start:end, start:end]
        avg_pae = np.mean(submatrix)
        ptm_score = 1 / (1 + (avg_pae / d0)) #d0 value is a parameter based on total protein lenght
        chain_ptm.append(ptm_score)
    return chain_ptm

def extract_pae_matrix(pae_data):   # can extract the actual pae matrix from a variety of formats
    if isinstance(pae_data, list) and len(pae_data) == 1 and isinstance(pae_data[0], dict):
        if 'predicted_aligned_error' in pae_data[0]:
            return np.array(pae_data[0]['predicted_aligned_error'])
        else:
            raise ValueError("Single dict in list does not have 'predicted_aligned_error' key.")
    if isinstance(pae_data, dict):
        if 'predicted_aligned_error' in pae_data:
            return np.array(pae_data['predicted_aligned_error'])
        elif len(pae_data) == 1:
            return np.array(list(pae_data.values())[0])
        else:
            raise ValueError("PAE dict format not recognized.")
    elif isinstance(pae_data, list):
        if isinstance(pae_data[0], list):
            return np.array(pae_data)
        else:
            length = int(np.sqrt(len(pae_data)))
            if length * length == len(pae_data):
                return np.array(pae_data).reshape((length, length))
            else:
                raise ValueError("PAE list format not recognized.")
    else:
        raise ValueError("PAE data format not recognized.")

def check_clashes_in_pdb(pdb_file, threshold=2.0):   # if two atoms are closer than 2 Å are considered clashes (uses both ATOM and HETATM lines in the .pdb)
    atoms = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    atoms.append((x, y, z))
                except ValueError:
                    continue
    for a1, a2 in itertools.combinations(atoms, 2):
        dx = a1[0] - a2[0]
        dy = a1[1] - a2[1]
        dz = a1[2] - a2[2]
        dist = (dx*dx + dy*dy + dz*dz) ** 0.5
        if dist < threshold:
            return 1.0
    return 0.0

def output_summary_confidence(script_dir, output_folder):   # puts together all the metrics: creates a file for each of the 5 models
    with open(os.path.join(script_dir, 'ranking_debug.json'), 'r') as f:
        ranking = json.load(f)
    iptm_ptm = ranking['iptm+ptm']
    iptm = ranking['iptm']
    for i in range(5):
        model_name = f'model_{i+1}_multimer_v3_pred_0'
        pkl_file = os.path.join(script_dir, f'result_{model_name}.pkl')
        with open(pkl_file, 'rb') as f:
            result = pickle.load(f)
        chain_iptm = []
        chain_pair_iptm = []
        chain_pair_pae_min = []
        chain_ptm = []
        plddt = result.get('plddt', np.array([]))
        if len(plddt) > 0:
            fraction_disordered = float((plddt < 50).sum()) / len(plddt)
        else:
            fraction_disordered = 0.0
        has_clash = 0.0
        num_recycles = int(result.get('num_recycles', 0))
        ptm = float(result.get('ptm', 0.0))
        iptm_val = float(result.get('iptm', iptm.get(model_name, 0.0)))
        ranking_score = float(result.get('ranking_confidence', iptm_ptm.get(model_name, 0.0)))
        pdb_file = os.path.join(script_dir, f'ranked_{i}.pdb')
        chain_ranges, chain_order = get_chain_residue_ranges_from_pdb(pdb_file)
        pae = result.get('predicted_aligned_error', None)
        if pae is not None and len(chain_order) > 0:
            chain_pair_pae_min = compute_chain_pair_pae_min(pae, chain_ranges, chain_order)
        else:
            chain_pair_pae_min = []
        confidence_pattern = os.path.join(script_dir, f'confidence_{model_name}*.json')
        confidence_files = glob.glob(confidence_pattern)
        chain_iptm = []
        if confidence_files:
            with open(confidence_files[0], 'r') as cf:
                conf_data = json.load(cf)
            confidence_scores = conf_data.get('confidenceScore', [])
            for chain in chain_order:
                start, end = chain_ranges[chain]
                avg_score = float(np.mean(confidence_scores[start:end]))
                chain_iptm.append(avg_score / 100)
        pae_pattern = os.path.join(script_dir, f'pae_{model_name}*.json')
        pae_files = glob.glob(pae_pattern)
        chain_ptm = []
        if pae_files:
            with open(pae_files[0], 'r') as pf:
                pae_data = json.load(pf)
            pae_matrix = extract_pae_matrix(pae_data)
            chain_ptm = compute_chain_ptm_from_pae(pae_matrix, chain_ranges, chain_order)
        chain_pair_iptm = []
        n_chains = len(chain_order)
        if n_chains > 0 and len(chain_iptm) == n_chains and len(chain_ptm) == n_chains:
            for chain_i in range(n_chains):
                row = []
                for chain_j in range(n_chains):
                    if chain_i == chain_j:
                        row.append(chain_ptm[chain_i])
                    else:
                        row.append(chain_iptm[chain_j])
                chain_pair_iptm.append(row)
        summary = {
            "chain_iptm": chain_iptm,
            "chain_pair_iptm": chain_pair_iptm,
            "chain_pair_pae_min": chain_pair_pae_min,
            "chain_ptm": chain_ptm,
            "fraction_disordered": fraction_disordered,
            "has_clash": has_clash,
            "iptm": iptm_val,
            "num_recycles": num_recycles,
            "ptm": ptm,
            "ranking_score": ranking_score
        }
        summary = round_floats(summary, 2)  
        unrelaxed_pdb_file = os.path.join(script_dir, f'unrelaxed_model_{i+1}_multimer_v3_pred_0.pdb')
        has_clash = check_clashes_in_pdb(unrelaxed_pdb_file, threshold=2.0)
        print(f"Writing summary_confidences: {os.path.join(output_folder, f'run_out_summary_confidences_{i}.json')}")
        with open(os.path.join(output_folder, f'run_out_summary_confidences_{i}.json'), 'w') as out:
            json.dump(convert(summary), out, indent=1)
