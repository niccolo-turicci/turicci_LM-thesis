import os
import json

# --- 5 - Creates job_request.json file: needed for AlphaFold and other modeling jobs (in our case for ABridge). ---
def extract_sequence_from_pdb(pdb_path): # it looks inside the .pdb for reconstructing the aminoacid sequence
    chains = {}
    seen = {}
    with open(pdb_path) as f: # as for the full_data it looks inside the .pdb file at specific positions in each ATOM line
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
    # ---------------------------------
    
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
    # Group chains by sequence to handle oligomers (e.g., dimers, tetramers)
    sequence_groups = {}  # sequence -> count
    for chain_id in sorted(chains.keys()):  
        if chains[chain_id]:  # makes sure chains have residues
            seq = ''.join(chains[chain_id])
            if seq not in sequence_groups:
                sequence_groups[seq] = 0
            sequence_groups[seq] += 1
    
    # Create sequences list with proper stoichiometry (matching AF3 format)
    sequences = []
    for seq, count in sequence_groups.items():
        sequences.append({
            "proteinChain": {
                "sequence": seq,
                "count": count,
                "useStructureTemplate": useStructureTemplate
            }
        })
    
    # Fills the fuelds of the file 
    job_request = [{
        "name": name,
        "modelSeeds": modelSeeds,
        "sequences": sequences,
        "dialect": dialect,
        "version": version
    }]
    
    with open(os.path.join(output_folder, "run_out_job_request.json"), "w") as f:
        json.dump(job_request, f, indent=1)
    
    total_chains = sum(seq_info["proteinChain"]["count"] for seq_info in sequences)
    print(f"job_request.json written to output_folder/run_out_job_request.json")
    for i, seq_info in enumerate(sequences):
        seq_len = len(seq_info["proteinChain"]["sequence"])
        count = seq_info["proteinChain"]["count"]
        print(f"  Sequence {i+1}: {seq_len} residues × {count} copies")
