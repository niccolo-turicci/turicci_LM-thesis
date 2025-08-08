import os
import json

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

def output_job_request(script_dir, output_folder, jobname):
    pdb_file = os.path.join(script_dir, "ranked_0.pdb")
    if not os.path.isfile(pdb_file):
        print(f"Error: {pdb_file} not found in selected directory.")
        return
    
    # heading (jobname = --jobname)
    name = jobname 
    # --- TO BE MODIFIED (manually) --- 
    modelSeeds = ["1234"]
    useStructureTemplate = True
    dialect = "alphafoldserver" # or AF3 (as dialect)
    version = 1
    # -----------------------
    
    # Converts the sequences to single letter (is what AB wants)
    aa_three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    chains = {}
    seen = {}
    with open(pdb_file) as f:
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
                if resn not in ["HOH", "WAT", "H2O"] and resn in aa_three_to_one:
                    chains[chain].append(aa_three_to_one[resn])

    # Create the job request file structure
    sequences = []
    for chain_id in sorted(chains.keys()):  
        if chains[chain_id]:  # makes sure chains have residues
            seq = ''.join(chains[chain_id])
            sequences.append({
                "proteinChain": {
                    "sequence": seq,
                    "count": 1,
                    "useStructureTemplate": useStructureTemplate
                }
            })
    
    # Actually creates the file 
    job_request = [{
        "name": name,
        "modelSeeds": modelSeeds,
        "sequences": sequences,
        "dialect": dialect,
        "version": version
    }]
    
    with open(os.path.join(output_folder, "run_out_job_request.json"), "w") as f:
        json.dump(job_request, f, indent=1)
    print("job_request.json (with {len(sequences)} protein chains) written to output_folder/run_out_job_request.json")
    for i, seq_info in enumerate(sequences):
        seq_len = len(seq_info["proteinChain"]["sequence"])
        print(f"  Chain {i+1}: {seq_len} residues")
