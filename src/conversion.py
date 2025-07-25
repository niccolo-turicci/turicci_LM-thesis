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


# --- 1 - Calculate pairwise residue distance: for each couple of C-alphas. Saves distances in a distance matrix. ---
def calculate_CA_distances(pdb_path, output_name):   # looks inside the .pdb file for all C-alpha
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    residues_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    residues_list.append((chain.get_id(), residue.get_id()[1]))
    N = len(residues_list)
    print(f"Total residues found: {N}")
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
    np.save(output_name, dist_matrix)  # matrix saved as a .npy file
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

# --- 2.1 - Calculates contact proability based on proximity (distance_matrix), model confidence (plddt), pae (pae_matrix). ---
def compute_contact_probabilities(distance_matrix, pae_matrix, plddt, threshold=8.0):
    plddt_norm = np.clip(plddt / 100.0, 0, 1)   # transforms plddt from 1-100 values to 0-1 values
    pairwise_plddt = (plddt_norm[:, None] + plddt_norm[None, :]) / 2.0   # calculates plddt mean for each residue pair
    pae_conf = expit(-(pae_matrix - threshold) / 1.5)   # using a sigmoid function converts pae score into a confidence score: treshold for contact is set at 8 Å
    within_threshold = (distance_matrix <= threshold).astype(float)   
    contact_probs = within_threshold * pae_conf * pairwise_plddt   # combines the confidence values into a single value for each res pair: returns a matrix called contact_probs
    return contact_probs
    
# --- 2.2 - Calculates contact proability based on .npy and .pkl files (model confidence files), by calling compute_contact_probabilities function  ---
def output_contact_probabilities(script_dir, temp_folder):
    npy_files = sorted(glob.glob(os.path.join(temp_folder, "distance_matrix_ranked_*.npy")))
    pkl_files = sorted(glob.glob(os.path.join(script_dir, "result_model_*_*.pkl")))
    if not npy_files or not pkl_files:
        print("No .npy or .pkl files found in the temp or input directory.")
        return
    for i in range(5):   # loops over all the 5 models produced by APD
        npy_pattern = f"distance_matrix_ranked_{i}.npy"
        pkl_pattern = f"result_model_{i+1}_*.pkl"
        npy_path = os.path.join(temp_folder, npy_pattern)
        pkl_candidates = glob.glob(os.path.join(script_dir, pkl_pattern))
        if not os.path.exists(npy_path) or not pkl_candidates:
            print(f"Skipping: {npy_pattern} or {pkl_pattern} not found.")
            continue
        pkl_path = pkl_candidates[0]
        print(f"Processing {npy_path} with {os.path.basename(pkl_path)}")
        distance_matrix = np.load(npy_path)
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
        plddt = pkl_data['plddt']
        pae_matrix = pkl_data['predicted_aligned_error']
        contact_probs = compute_contact_probabilities(distance_matrix, pae_matrix, plddt)
        output_base = os.path.join(temp_folder, f"contact_probs_ranked_{i}")
        np.save(output_base + ".npy", contact_probs)
        np.savetxt(output_base + ".csv", contact_probs, delimiter=",")
        print(f"Output saved as {output_base}.npy and {output_base}.csv in temp_files folder")

# --- 3 - Creates the full_data.json file collecting info from different confidence files. ---
def load_pae_json(json_path):   # takes the .json file containing the pae matrix looking for the line with the pae data
    with open(json_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if "predicted_aligned_error" in first:
                return first["predicted_aligned_error"]
        raise ValueError("PAE matrix not found in JSON structure.")

def parse_pdb(pdb_path):   # looks for chain IDs, plddt scores (for each atom), residue chain IDs and residue numbers inside the .pdb file
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

def validate_output_json(filepath):   # not actually necessary: can be commented out 
    expected_keys = {
        "atom_chain_ids",
        "atom_plddts",
        "contact_probs",
        "pae",
        "token_chain_ids",
        "token_res_ids"
    }
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
            print(".json file structure is valid.")
    except Exception as e:
        print(f"Error in .json file structure: {e}")

def output_full_data(script_dir, temp_folder, output_folder):   # actually creates the full_data.json; for each model
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
        print(f"Processing:\n  PDB: {os.path.basename(pdb_path)}\n  PAE: {os.path.basename(pae_path)}\n  CONTACT: {os.path.basename(contact_probs_path)}")
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
        output_filename = os.path.join(output_folder, f"run_out_full_data_{i}.json")
        with open(output_filename, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f".json file saved to {output_filename}")
        validate_output_json(output_filename)

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
    chain_ptm = []
    for chain in chain_order:
        start, end = chain_ranges[chain]
        submatrix = pae_matrix[start:end, start:end]
        avg_pae = np.mean(submatrix)
        ptm_score = 1 / (1 + (avg_pae / 31))
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
        print(f"Writing: {os.path.join(output_folder, f'run_out_summary_confidences_{i}.json')}")
        with open(os.path.join(output_folder, f'run_out_summary_confidences_{i}.json'), 'w') as out:
            json.dump(convert(summary), out, indent=1)

# --- 5 - Creates job_request.json file: needed for AlphaFold and other modeling jobs. ---
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
    dialect = "alphafoldserver" # or AF3 (as dialect)
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
    with open(os.path.join(output_folder, "run_out_job_request.json"), "w") as f:
        json.dump(job_request, f, indent=1)
    print("Job request written to output_folder/run_out_job_request.json")

# --- 6 - Converts the .pdb files into .cif files to be fed into AlphaBridge ---
def convert_pdb_to_cif(script_dir, output_folder):
    pdb_files = sorted(glob.glob(os.path.join(script_dir, "ranked_*.pdb")))
    if not pdb_files:
        print("No ranked_*.pdb files found for conversion.")
        return

    # Three-letter to one-letter  conversion
    aa_three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    for pdb_file in pdb_files:
        structure_id = os.path.splitext(os.path.basename(pdb_file))[0].replace("ranked_", "")
        cif_file = os.path.join(output_folder, f"run_out_model_{structure_id}.cif")
        
        # Extract information
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        chains_info = {}
        entity_id = 1
        
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                if chain_id not in chains_info:
                    chains_info[chain_id] = {
                        'entity_id': entity_id,
                        'sequence': [],
                        'residues': []
                    }
                    entity_id += 1
                
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # The standard residue
                        res_name = residue.get_resname()
                        res_num = residue.get_id()[1]
                        if res_name in aa_three_to_one:
                            chains_info[chain_id]['sequence'].append(res_name)
                            chains_info[chain_id]['residues'].append((res_name, res_num))
        
        # Actually write the .cif file: AlphaFold 3 format structure
        with open(cif_file, 'w') as f:
            
            # Header
            f.write("This file was generated within the pipeline for converting APD output into AB input\n")
            f.write(f"_entry.id {structure_id}_converted\n")
            f.write("#\n")
            
            # Atom types
            f.write("loop_\n")
            f.write("_atom_type.symbol\n")
            f.write("C \nN \nO \nS \n")
            f.write("#\n")
            
            # aa definitions (header)
            f.write("loop_\n")
            f.write("_chem_comp.formula\n")
            f.write("_chem_comp.formula_weight\n")
            f.write("_chem_comp.id\n")
            f.write("_chem_comp.mon_nstd_flag\n")
            f.write("_chem_comp.name\n")
            f.write("_chem_comp.pdbx_smiles\n")
            f.write("_chem_comp.pdbx_synonyms\n")
            f.write("_chem_comp.type\n")
            # aa definitions (here all the informations about the standard aminoacids are stored. formula, weight, full name, smiles)
            aa_definitions = {
                'ALA': ('"C3 H7 N O2"', '89.093', 'ALANINE', 'C[C@H](N)C(O)=O'),
                'ARG': ('"C6 H15 N4 O2"', '175.209', 'ARGININE', 'N[C@@H](CCCNC(N)=[NH2+])C(O)=O'),
                'ASN': ('"C4 H8 N2 O3"', '132.118', 'ASPARAGINE', 'N[C@@H](CC(N)=O)C(O)=O'),
                'ASP': ('"C4 H7 N O4"', '133.103', '"ASPARTIC ACID"', 'N[C@@H](CC(O)=O)C(O)=O'),
                'CYS': ('"C3 H7 N O2 S"', '121.158', 'CYSTEINE', 'N[C@@H](CS)C(O)=O'),
                'GLN': ('"C5 H10 N2 O3"', '146.144', 'GLUTAMINE', 'N[C@@H](CCC(N)=O)C(O)=O'),
                'GLU': ('"C5 H9 N O4"', '147.129', '"GLUTAMIC ACID"', 'N[C@@H](CCC(O)=O)C(O)=O'),
                'GLY': ('"C2 H5 N O2"', '75.067', 'GLYCINE', 'NCC(O)=O'),
                'HIS': ('"C6 H10 N3 O2"', '156.162', 'HISTIDINE', 'N[C@@H](Cc1c[nH]c[nH+]1)C(O)=O'),
                'ILE': ('"C6 H13 N O2"', '131.173', 'ISOLEUCINE', 'CC[C@H](C)[C@H](N)C(O)=O'),
                'LEU': ('"C6 H13 N O2"', '131.173', 'LEUCINE', 'CC(C)C[C@H](N)C(O)=O'),
                'LYS': ('"C6 H15 N2 O2"', '147.195', 'LYSINE', 'N[C@@H](CCCC[NH3+])C(O)=O'),
                'MET': ('"C5 H11 N O2 S"', '149.211', 'METHIONINE', 'CSCC[C@H](N)C(O)=O'),
                'PHE': ('"C9 H11 N O2"', '165.189', 'PHENYLALANINE', 'N[C@@H](Cc1ccccc1)C(O)=O'),
                'PRO': ('"C5 H9 N O2"', '115.130', 'PROLINE', 'OC(=O)[C@@H]1CCCN1'),
                'SER': ('"C3 H7 N O3"', '105.093', 'SERINE', 'N[C@@H](CO)C(O)=O'),
                'THR': ('"C4 H9 N O3"', '119.119', 'THREONINE', 'C[C@@H](O)[C@H](N)C(O)=O'),
                'TRP': ('"C11 H12 N2 O2"', '204.225', 'TRYPTOPHAN', 'N[C@@H](Cc1c[nH]c2ccccc12)C(O)=O'),
                'TYR': ('"C9 H11 N O3"', '181.189', 'TYROSINE', 'N[C@@H](Cc1ccc(O)cc1)C(O)=O'),
                'VAL': ('"C5 H11 N O2"', '117.146', 'VALINE', 'CC(C)[C@H](N)C(O)=O')
            }
            # Gets aa from actual sequence: it creates a set (deletes duplicates), and writes every unique aminoacid that finds
            unique_aas = set()
            for chain_data in chains_info.values():
                unique_aas.update(chain_data['sequence'])
            
            for aa in sorted(unique_aas):
                if aa in aa_definitions:
                    formula, weight, name, smiles = aa_definitions[aa]
                    f.write(f'{formula}    {weight}  {aa} y {name}         {smiles}                  ? "L-PEPTIDE LINKING" \n')
            f.write("#\n")

            # Defines the entities (chains) of the file
            f.write("loop_\n")
            f.write("_entity.id\n")
            f.write("_entity.pdbx_description\n")
            f.write("_entity.type\n")
            for chain_id in sorted(chains_info.keys()):
                entity_id = chains_info[chain_id]['entity_id']
                f.write(f"{entity_id} . polymer \n")
            f.write("#\n")

            # Defines the entities (chains) of the file as before: more precisely this time
            f.write("loop_\n")
            f.write("_entity_poly.entity_id\n")
            f.write("_entity_poly.pdbx_strand_id\n")
            f.write("_entity_poly.type\n")
            for chain_id in sorted(chains_info.keys()):
                entity_id = chains_info[chain_id]['entity_id']
                f.write(f"{entity_id} {chain_id} polypeptide(L) \n")
            f.write("#\n")
            
           # For each aa in the sequence, it writes the entity ID, "n" (this is not a hetatm)", aa code, and the position 
            f.write("loop_\n")
            f.write("_entity_poly_seq.entity_id\n")
            f.write("_entity_poly_seq.hetero\n")
            f.write("_entity_poly_seq.mon_id\n")
            f.write("_entity_poly_seq.num\n")
            for chain_id in sorted(chains_info.keys()):
                entity_id = chains_info[chain_id]['entity_id']
                sequence = chains_info[chain_id]['sequence']
                for i, aa in enumerate(sequence, 1):
                    f.write(f"{entity_id} n {aa} {i}    \n")
            f.write("#\n")
            
            
            f.write("_ma_data.content_type \"model coordinates\"\n")
            f.write("_ma_data.id           1\n")
            f.write("_ma_data.name         Model\n")
            f.write("#\n")
            
            
            f.write("_ma_model_list.data_id          1\n")
            f.write("_ma_model_list.model_group_id   1\n")
            f.write("_ma_model_list.model_group_name \"AlphaFold model\"\n")
            f.write("_ma_model_list.model_id         1\n")
            f.write("_ma_model_list.model_name       \"Converted model\"\n")
            f.write("_ma_model_list.model_type       \"Ab initio model\"\n")
            f.write("_ma_model_list.ordinal_id       1\n")
            f.write("#\n")
            
            
            f.write("loop_\n")
            f.write("_ma_protocol_step.method_type\n")
            f.write("_ma_protocol_step.ordinal_id\n")
            f.write("_ma_protocol_step.protocol_id\n")
            f.write("_ma_protocol_step.step_id\n")
            f.write("\"coevolution MSA\" 1 1 1 \n")
            f.write("\"template search\" 2 1 2 \n")
            f.write("modeling          3 1 3 \n")
            f.write("#\n")
            
            
            f.write("loop_\n")
            f.write("_ma_qa_metric.id\n")
            f.write("_ma_qa_metric.mode\n")
            f.write("_ma_qa_metric.name\n")
            f.write("_ma_qa_metric.software_group_id\n")
            f.write("_ma_qa_metric.type\n")
            f.write("1 global pLDDT 1 pLDDT \n")
            f.write("2 local  pLDDT 1 pLDDT \n")
            f.write("#\n")
            
            
            f.write("_ma_qa_metric_global.metric_id    1\n")
            f.write("_ma_qa_metric_global.metric_value 75.00\n")
            f.write("_ma_qa_metric_global.model_id     1\n")
            f.write("_ma_qa_metric_global.ordinal_id   1\n")
            f.write("#\n")
            
            
            f.write("_ma_software_group.group_id    1\n")
            f.write("_ma_software_group.ordinal_id  1\n")
            f.write("_ma_software_group.software_id 1\n")
            f.write("#\n")
            
            
            f.write("loop_\n")
            f.write("_ma_target_entity.data_id\n")
            f.write("_ma_target_entity.entity_id\n")
            f.write("_ma_target_entity.origin\n")
            for chain_id in sorted(chains_info.keys()):
                entity_id = chains_info[chain_id]['entity_id']
                f.write(f"1 {entity_id} . \n")
            f.write("#\n")
            
            
            f.write("loop_\n")
            f.write("_ma_target_entity_instance.asym_id\n")
            f.write("_ma_target_entity_instance.details\n")
            f.write("_ma_target_entity_instance.entity_id\n")
            for chain_id in sorted(chains_info.keys()):
                entity_id = chains_info[chain_id]['entity_id']
                f.write(f"{chain_id} . {entity_id} \n")
            f.write("#\n")
            
            
            f.write("loop_\n")
            f.write("_pdbx_data_usage.details\n")
            f.write("_pdbx_data_usage.id\n")
            f.write("_pdbx_data_usage.type\n")
            f.write("_pdbx_data_usage.url\n")
            f.write(";NON-COMMERCIAL USE ONLY, BY USING THIS FILE YOU AGREE TO THE TERMS OF USE FOUND\n")
            f.write("AT https://alphafoldserver.com/output-terms\n")
            f.write("; 1 \"Terms of use\" https://alphafoldserver.com/output-terms \n")
            f.write("#\n")
            
            
            f.write("_software.classification other\n")
            f.write("_software.date           ?\n")
            f.write("_software.description    \"Structure prediction\"\n")
            f.write("_software.name           AlphaFold\n")
            f.write("_software.pdbx_ordinal   1\n")
            f.write("_software.type           package\n")
            f.write("_software.version        \"Converted from PDB\"\n")
            f.write("#\n")
            
            
            f.write("loop_\n")
            f.write("_struct_asym.entity_id\n")
            f.write("_struct_asym.id\n")
            for chain_id in sorted(chains_info.keys()):
                entity_id = chains_info[chain_id]['entity_id']
                f.write(f"{entity_id} {chain_id} \n")
            f.write("#\n")

            # The middle section
            f.write("loop_\n")
            f.write("_pdbx_poly_seq_scheme.asym_id\n")
            f.write("_pdbx_poly_seq_scheme.auth_seq_num\n")
            f.write("_pdbx_poly_seq_scheme.entity_id\n")
            f.write("_pdbx_poly_seq_scheme.hetero\n")
            f.write("_pdbx_poly_seq_scheme.mon_id\n")
            f.write("_pdbx_poly_seq_scheme.pdb_ins_code\n")
            f.write("_pdbx_poly_seq_scheme.pdb_seq_num\n")
            f.write("_pdbx_poly_seq_scheme.pdb_strand_id\n")
            f.write("_pdbx_poly_seq_scheme.seq_id\n")
            for chain_id in sorted(chains_info.keys()):
                entity_id = chains_info[chain_id]['entity_id']
                sequence = chains_info[chain_id]['sequence']
                residues = chains_info[chain_id]['residues']
                for i, (aa, res_num) in enumerate(residues, 1):
                    f.write(f"{chain_id} {res_num}   {entity_id} n {aa} . {res_num}   {chain_id} {i}    \n")
            f.write("#\n")


            # Write the atomic coordinates using MMCIFIO (biopython's modeule that hanldes mmCIF files):

            # Temporary CIF file to get the atom_site section
            temp_cif = cif_file + ".temp"
            io_cif = PDB.MMCIFIO()
            io_cif.set_structure(structure)
            io_cif.save(temp_cif)

            # Read coordinates from the temp file
            with open(temp_cif, 'r') as temp_f:
                temp_content = temp_f.read()
            
                if '_atom_site.' in temp_content:
                    atom_section_start = temp_content.find('loop_\n_atom_site.')
                    if atom_section_start != -1:
                        atom_section = temp_content[atom_section_start:]
                        f.write(atom_section)
            
            # Delete temp file
            if os.path.exists(temp_cif):
                os.remove(temp_cif)
        
        print(f"Converted {pdb_file} to {cif_file} with proper format")
