import os
import json
import glob
import numpy as np
import pickle
from scipy.special import expit
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
    print(f"Distance matrix saved as {output_name} in temp_files folder.")

def output_residue_distances(script_dir, temp_folder):
    pdb_files = [f for f in os.listdir(script_dir) if f.endswith('.pdb') and f.startswith('ranked')] # only looks for "ranked_X.pdb" structures
    if not pdb_files:
        print("no .pdb files found")
    else:
        for pdb_file in pdb_files:
            pdb_path = os.path.join(script_dir, pdb_file)
            output_name = os.path.join(temp_folder, f"distance_matrix_{os.path.splitext(pdb_file)[0]}.npy")
            print(f"Processing {pdb_path} structure ...")
            calculate_CA_distances(pdb_path, output_name)

# --- 2.1 - Create the function to calculate contact proability based on proximity (distance_matrix), model confidence (plddt), pae (pae_matrix). ---
def compute_contact_probabilities(distance_matrix, pae_matrix, plddt, threshold=8.0): 
    plddt_norm = np.clip(plddt / 100.0, 0, 1)   # transforms plddt from 1-100 values to 0-1 values
    pairwise_plddt = np.sqrt(plddt_norm[:, None] * plddt_norm[None, :])   # calculates plddt geometric mean for each residue pair
    pae_conf = expit(-(pae_matrix - threshold) / 1.5)   # using a sigmoid function converts pae score into a confidence score: treshold for possible contact is set at 8 Å
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
    for i in range(5):   # loops over all the 5 models produced by APD (ranked_X.pdb)
        npy_pattern = f"distance_matrix_ranked_{i}.npy"
        pkl_pattern = f"result_model_{i+1}_*.pkl"
        npy_path = os.path.join(temp_folder, npy_pattern)
        pkl_candidates = glob.glob(os.path.join(script_dir, pkl_pattern))
        if not os.path.exists(npy_path) or not pkl_candidates:
            print(f"Skipping: {npy_pattern} or {pkl_pattern} not found.")
            continue
        pkl_path = pkl_candidates[0]
        print(f"Calcultaing contact probability with {npy_path} and {os.path.basename(pkl_path)}")
        distance_matrix = np.load(npy_path)
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
        plddt = pkl_data['plddt']
        pae_matrix = pkl_data['predicted_aligned_error']
        contact_probs = compute_contact_probabilities(distance_matrix, pae_matrix, plddt)
        output_base = os.path.join(temp_folder, f"contact_probs_ranked_{i}")
        np.save(output_base + ".npy", contact_probs)
        np.savetxt(output_base + ".csv", contact_probs, delimiter=",")
        print(f"Contact prob. matrix saved as {output_base}.npy and {output_base}.csv in temp_files folder")

# --- 3 - Creates the full_data.json file collecting info from different confidence files. ---
def load_pae_json(json_path):   # takes the .json file containing the pae matrix looking for the line with the pae data
    with open(json_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if "predicted_aligned_error" in first:
                return first["predicted_aligned_error"]
        raise ValueError("PAE matrix not found in JSON file.")

def parse_pdb(pdb_path):   # looks for chain IDs, plddt scores (for each atom), residue chain IDs and residue numbers inside the .pdb file
    atom_chain_ids = []
    atom_plddts = []
    token_chain_ids = []
    token_res_ids = []
    seen_residues = set()
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"): # looks inside the lines with ATOM (ignores HETATM)
                chain_id = line[21].strip() # chain_id is found at position 21
                res_id = int(line[22:26].strip()) # residue_id is found at position 22-26
                b_factor = float(line[60:66].strip()) # b-factor (our pLDDT) is found at position 60-66
                atom_chain_ids.append(chain_id)
                atom_plddts.append(b_factor)
                res_key = (chain_id, res_id)
                if res_key not in seen_residues:
                    token_chain_ids.append(chain_id)
                    token_res_ids.append(res_id)
                    seen_residues.add(res_key)
    return atom_chain_ids, atom_plddts, token_chain_ids, token_res_ids

def validate_output_json(filepath):   # Checks if the final file has a valid structure: not actually necessary, can be commented out 
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

def output_full_data(script_dir, temp_folder, output_folder):   # actually creates the full_data.json; collects all the computed info for each model
    pdb_files = sorted(glob.glob(os.path.join(script_dir, "ranked_*.pdb")))
    pae_files = sorted(glob.glob(os.path.join(script_dir, "pae_model_*.json")))
    contact_files = sorted(glob.glob(os.path.join(temp_folder, "contact_probs_ranked_*.npy")))
    if not (len(pdb_files) == len(pae_files) == len(contact_files) == 5):
        print("Error: Expected 5 of each file type (ranked_*.pdb, pae_model_*.json, contact_probs_ranked_*.npy).")
        print(f"Found: {len(pdb_files)} for the structure in .pdb format, {len(pae_files)} for pae values, {len(contact_files)} as contact files.")
        return
    for i in range(5):
        pdb_path = pdb_files[i]
        pae_path = pae_files[i]
        contact_probs_path = contact_files[i]
        print(f"Creating full_data using:\n  as PDB: {os.path.basename(pdb_path)}\n  as PAE matrix : {os.path.basename(pae_path)}\n  for CONTACT PROB: {os.path.basename(contact_probs_path)}")
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
        print(f"fulla_data.json file saved to {output_filename}")
        validate_output_json(output_filename)
