import os
import json
import pickle
import numpy as np
import math
import re
import glob
import itertools
import csv
import copy
from scipy.special import expit  # sigmoid
from Bio import PDB

# --- 0- define input and output folders ---
def get_working_directory():
    folder = input("Enter the path to the folder to process: ").strip()
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.")
        exit(1)
    return folder

def get_output_directories():
    base_name = input("Enter the name for the output folder: ").strip()
    base_folder = os.path.join(os.getcwd(), base_name)
    temp_folder = os.path.join(base_folder, "temp_files")
    output_folder = os.path.join(base_folder, "output_folder")
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    return temp_folder, output_folder

# --- 1 - calculate residue distances ---
def calculate_CA_distances(pdb_path, output_name):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    residues_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    residues_list.append((chain.get_id(), residue.get_id()[1]))
    N = len(residues_list)
    print(f"Total residues with CA: {N}")
    dist_matrix = np.full((N, N), np.inf)
    residue_index = {res: i for i, res in enumerate(residues_list)}
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" not in residue:
                    continue
                atom1 = residue["CA"]
                idx1 = residue_index[(chain.get_id(), residue.get_id()[1])]
                for chain2 in model:
                    for residue2 in chain2:
                        if "CA" not in residue2:
                            continue
                        atom2 = residue2["CA"]
                        idx2 = residue_index[(chain2.get_id(), residue2.get_id()[1])]
                        dist = atom1 - atom2
                        dist_matrix[idx1, idx2] = dist
    np.save(output_name, dist_matrix)
    print(f"Distance matrix saved as {output_name}")

def output_residue_distances(script_dir, temp_folder):
    pdb_files = [f for f in os.listdir(script_dir) if f.endswith('.pdb') and f.startswith('ranked')]
    if not pdb_files:
        print("no .pdb files found")
    else:
        for pdb_file in pdb_files:
            pdb_path = os.path.join(script_dir, pdb_file)
            output_name = os.path.join(temp_folder, f"distance_matrix_{os.path.splitext(pdb_file)[0]}.npy")
            print(f"Processing {pdb_path} ...")
            calculate_CA_distances(pdb_path, output_name)

# --- 2 - calculate contact proability ---
def compute_contact_probabilities(distance_matrix, pae_matrix, plddt, threshold=8.0):
    plddt_norm = np.clip(plddt / 100.0, 0, 1)
    pairwise_plddt = (plddt_norm[:, None] + plddt_norm[None, :]) / 2.0
    pae_conf = expit(-(pae_matrix - threshold) / 1.5)
    within_threshold = (distance_matrix <= threshold).astype(float)
    contact_probs = within_threshold * pae_conf * pairwise_plddt
    return contact_probs

def output_contact_probabilities(script_dir, temp_folder):
    npy_files = sorted(glob.glob(os.path.join(temp_folder, "distance_matrix_ranked_*.npy")))
    pkl_files = sorted(glob.glob(os.path.join(script_dir, "result_model_*_*.pkl")))
    if not npy_files or not pkl_files:
        print("No .npy or .pkl files found in the temp or input directory.")
        return
    for i in range(5):
        npy_pattern = f"distance_matrix_ranked_{i}.npy"
        pkl_pattern = f"result_model_{i+1}_*.pkl"
        npy_path = os.path.join(temp_folder, npy_pattern)
        pkl_candidates = glob.glob(os.path.join(script_dir, pkl_pattern))
        if not os.path.exists(npy_path) or not pkl_candidates:
            print(f"Skipping: {npy_pattern} or {pkl_pattern} not found.")
            continue
        pkl_path = pkl_candidates[0]
        print(f"[INFO] Processing {npy_path} with {os.path.basename(pkl_path)}")
        distance_matrix = np.load(npy_path)
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
        plddt = pkl_data['plddt']
        pae_matrix = pkl_data['predicted_aligned_error']
        contact_probs = compute_contact_probabilities(distance_matrix, pae_matrix, plddt)
        output_base = os.path.join(temp_folder, f"contact_probs_ranked_{i}")
        np.save(output_base + ".npy", contact_probs)
        np.savetxt(output_base + ".csv", contact_probs, delimiter=",")
        print(f"[DONE] Output saved as {output_base}.npy and {output_base}.csv")

# --- 3 - create the full_data.json ---
def load_pae_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if "predicted_aligned_error" in first:
                return first["predicted_aligned_error"]
        raise ValueError("PAE matrix not found in JSON structure.")

def parse_pdb(pdb_path):
    atom_chain_ids = []
    atom_plddts = []
    token_chain_ids = []
    token_res_ids = []
    seen_residues = set()
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain_id = line[21].strip()
                res_id = int(line[22:26].strip())
                b_factor = float(line[60:66].strip())
                atom_chain_ids.append(chain_id)
                atom_plddts.append(b_factor)
                res_key = (chain_id, res_id)
                if res_key not in seen_residues:
                    token_chain_ids.append(chain_id)
                    token_res_ids.append(res_id)
                    seen_residues.add(res_key)
    return atom_chain_ids, atom_plddts, token_chain_ids, token_res_ids

def validate_output_json(filepath):
    expected_keys = {
        "atom_chain_ids",
        "atom_plddts",
        "contact_probs",
        "pae",
        "token_chain_ids",
        "token_res_ids"
    }
    print(f"[VALIDATION] Checking {filepath}")
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            data_keys = set(data.keys())
            missing = expected_keys - data_keys
            if missing:
                raise ValueError(f"Missing keys in JSON: {missing}")
            for key in expected_keys:
                if not isinstance(data[key], list):
                    raise ValueError(f"Key '{key}' is not a list")
            print("[VALIDATION] File structure is valid.")
    except Exception as e:
        print(f"[VALIDATION] Error in JSON structure: {e}")

def output_full_data(script_dir, temp_folder, output_folder):
    pdb_files = sorted(glob.glob(os.path.join(script_dir, "ranked_*.pdb")))
    pae_files = sorted(glob.glob(os.path.join(script_dir, "pae_model_*.json")))
    contact_files = sorted(glob.glob(os.path.join(temp_folder, "contact_probs_ranked_*.npy")))
    if not (len(pdb_files) == len(pae_files) == len(contact_files) == 5):
        print("Error: Expected 5 of each file type (ranked_*.pdb, pae_model_*.json, contact_probs_ranked_*.npy).")
        print(f"Found: {len(pdb_files)} pdb, {len(pae_files)} pae, {len(contact_files)} contact files.")
        return
    for i in range(5):
        pdb_path = pdb_files[i]
        pae_path = pae_files[i]
        contact_probs_path = contact_files[i]
        print(f"[INFO] Processing:\n  PDB: {os.path.basename(pdb_path)}\n  PAE: {os.path.basename(pae_path)}\n  CONTACT: {os.path.basename(contact_probs_path)}")
        pae_matrix = load_pae_json(pae_path)
        contact_matrix = np.load(contact_probs_path).tolist()
        atom_chain_ids, atom_plddts, token_chain_ids, token_res_ids = parse_pdb(pdb_path)
        output_data = {
            "atom_chain_ids": atom_chain_ids,
            "atom_plddts": atom_plddts,
            "contact_probs": contact_matrix,
            "pae": pae_matrix,
            "token_chain_ids": token_chain_ids,
            "token_res_ids": token_res_ids
        }
        pdb_base = os.path.splitext(os.path.basename(pdb_path))[0]
        output_filename = os.path.join(output_folder, f"{pdb_base}_full_data.json")
        with open(output_filename, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"[DONE] JSON saved to {output_filename}")
        validate_output_json(output_filename)

# --- 4 - create summary_confidence.json ---
def convert(obj):
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

def round_floats(obj, ndigits=2):
    if isinstance(obj, float):
        return round(obj, ndigits)
    elif isinstance(obj, list):
        return [round_floats(i, ndigits) for i in obj]
    elif isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    else:
        return obj

def get_chain_residue_ranges_from_pdb(pdb_file):
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

def compute_chain_pair_pae_min(pae, chain_ranges, chain_order):
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

def compute_chain_ptm_from_pae(pae_matrix, chain_ranges, chain_order):
    chain_ptm = []
    for chain in chain_order:
        start, end = chain_ranges[chain]
        submatrix = pae_matrix[start:end, start:end]
        avg_pae = np.mean(submatrix)
        ptm_score = 1 / (1 + (avg_pae / 31))
        chain_ptm.append(ptm_score)
    return chain_ptm

def extract_pae_matrix(pae_data):
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

def check_clashes_in_pdb(pdb_file, threshold=2.0):
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

def output_summary_confidence(script_dir, output_folder):
    with open(os.path.join(script_dir, 'ranking_debug.json'), 'r') as f:
        ranking = json.load(f)
    iptm_ptm = ranking['iptm+ptm']
    iptm = ranking['iptm']
    for i in range(1, 6):
        model_name = f'model_{i}_multimer_v3_pred_0'
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
        pdb_file = os.path.join(script_dir, f'ranked_{i-1}.pdb')
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
                chain_iptm.append(avg_score)
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
            for i in range(n_chains):
                row = []
                for j in range(n_chains):
                    if i == j:
                        row.append(chain_ptm[i])
                    else:
                        row.append(chain_iptm[j])
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
        unrelaxed_pdb_file = os.path.join(script_dir, f'unrelaxed_model_{i}_multimer_v3_pred_0.pdb')
        has_clash = check_clashes_in_pdb(unrelaxed_pdb_file, threshold=2.0)
        with open(os.path.join(output_folder, f'summary_confidences_{model_name}.json'), 'w') as out:
            json.dump(convert(summary), out, indent=1)

# --- 5 - create job_request.json ---
def extract_sequence_from_pdb(pdb_path):
    chains = {}
    seen = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21]
                resn = line[17:20].strip()
                resi = int(line[22:26])
                key = (chain, resi)
                if key in seen:
                    continue
                seen[key] = True
                if chain not in chains:
                    chains[chain] = []
                if resn not in ["HOH", "WAT", "H2O"]:
                    chains[chain].append(resn)
    seqs = {chain: ''.join(residues) for chain, residues in chains.items()}
    return seqs

def output_job_request(script_dir, output_folder):
    pdb_file = os.path.join(script_dir, "ranked_0.pdb")
    if not os.path.isfile(pdb_file):
        print(f"Error: {pdb_file} not found in selected directory.")
        return
    # --- TO BE MODIFIED (manually) --- 
    name = "name of the job"
    modelSeeds = ["1234"]
    useStructureTemplate = True
    dialect = "alphafoldserver"
    version = 1
    # -----------------------
    seqs = extract_sequence_from_pdb(pdb_file)
    sequences = []
    for chain, seq in seqs.items():
        sequences.append({
            "proteinChain": {
                "sequence": seq,
                "count": 1,
                "useStructureTemplate": useStructureTemplate
            }
        })
    job_request = [{
        "name": name,
        "modelSeeds": modelSeeds,
        "sequences": sequences,
        "dialect": dialect,
        "version": version
    }]
    with open(os.path.join(output_folder, "job_request.json"), "w") as f:
        json.dump(job_request, f, indent=1)
    print("Job request written to output_folder/job_request.json")

# --- 6 - converts the .pdb files into .cif files ---
def convert_pdb_to_cif(script_dir, output_folder):
    pdb_files = sorted(glob.glob(os.path.join(script_dir, "ranked_*.pdb")))
    if not pdb_files:
        print("No ranked_*.pdb files found for conversion.")
        return

    io_pdb = PDB.PDBParser(QUIET=True)
    io_cif = PDB.MMCIFIO()
    for pdb_file in pdb_files:
        structure_id = os.path.splitext(os.path.basename(pdb_file))[0]
        structure = io_pdb.get_structure(structure_id, pdb_file)
        cif_file = os.path.join(output_folder, f"{structure_id}.cif")
        io_cif.set_structure(structure)
        io_cif.save(cif_file)
        print(f"Converted {pdb_file} to {cif_file}")

# --- Unified main ---
def main():
    script_dir = get_working_directory()
    temp_folder, output_folder = get_output_directories()
    output_residue_distances(script_dir, temp_folder)
    output_contact_probabilities(script_dir, temp_folder)
    output_full_data(script_dir, temp_folder, output_folder)
    output_summary_confidence(script_dir, output_folder)
    output_job_request(script_dir, output_folder)
    convert_pdb_to_cif(script_dir, output_folder)

if __name__ == "__main__":
    main()
