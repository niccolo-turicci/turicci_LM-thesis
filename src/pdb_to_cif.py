import os
import glob
from Bio import PDB

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
            f.write(f"data_ {structure_id}\n ")
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
            
           # For each aa in the sequence, it writes the entity ID, "n" (this is not a HETATM), aa code, and the position 
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
            
            # data about coordinates
            f.write("_ma_data.content_type \"model coordinates\"\n")
            f.write("_ma_data.id           1\n")
            f.write("_ma_data.name         Model\n")
            f.write("#\n")
            
            # defines the model list: this is an AlphaFold model (ab initio)
            f.write("_ma_model_list.data_id          1\n")
            f.write("_ma_model_list.model_group_id   1\n")
            f.write("_ma_model_list.model_group_name \"AlphaFold model\"\n")
            f.write("_ma_model_list.model_id         1\n")
            f.write("_ma_model_list.model_name       \"Converted model\"\n")
            f.write("_ma_model_list.model_type       \"Ab initio model\"\n")
            f.write("_ma_model_list.ordinal_id       1\n")
            f.write("#\n")
            
            # what were the steps: MSA, template search, etc. 
            f.write("loop_\n")
            f.write("_ma_protocol_step.method_type\n")
            f.write("_ma_protocol_step.ordinal_id\n")
            f.write("_ma_protocol_step.protocol_id\n")
            f.write("_ma_protocol_step.step_id\n")
            f.write("\"coevolution MSA\" 1 1 1 \n")
            f.write("\"template search\" 2 1 2 \n")
            f.write("modeling          3 1 3 \n")
            f.write("#\n")
            
            # describe the pLDDT metrics
            f.write("loop_\n")
            f.write("_ma_qa_metric.id\n")
            f.write("_ma_qa_metric.mode\n")
            f.write("_ma_qa_metric.name\n")
            f.write("_ma_qa_metric.software_group_id\n")
            f.write("_ma_qa_metric.type\n")
            f.write("1 global pLDDT 1 pLDDT \n")
            f.write("2 local  pLDDT 1 pLDDT \n")
            f.write("#\n")
            
            # specify global quality (default to 75.00)
            f.write("_ma_qa_metric_global.metric_id    1\n")
            f.write("_ma_qa_metric_global.metric_value 75.00\n")
            f.write("_ma_qa_metric_global.model_id     1\n")
            f.write("_ma_qa_metric_global.ordinal_id   1\n")
            f.write("#\n")
            
            # software (AlphaFold) information
            f.write("_ma_software_group.group_id    1\n")
            f.write("_ma_software_group.ordinal_id  1\n")
            f.write("_ma_software_group.software_id 1\n")
            f.write("#\n")
            
            # defines which entities (chains) there are in the file
            f.write("loop_\n")
            f.write("_ma_target_entity.data_id\n")
            f.write("_ma_target_entity.entity_id\n")
            f.write("_ma_target_entity.origin\n")
            for chain_id in sorted(chains_info.keys()):
                entity_id = chains_info[chain_id]['entity_id']
                f.write(f"1 {entity_id} . \n")
            f.write("#\n")
            
            # links chain ID to entity ID
            f.write("loop_\n")
            f.write("_ma_target_entity_instance.asym_id\n")
            f.write("_ma_target_entity_instance.details\n")
            f.write("_ma_target_entity_instance.entity_id\n")
            for chain_id in sorted(chains_info.keys()):
                entity_id = chains_info[chain_id]['entity_id']
                f.write(f"{chain_id} . {entity_id} \n")
            f.write("#\n")
            
            # commercial-use information
            f.write("loop_\n")
            f.write("_pdbx_data_usage.details\n")
            f.write("_pdbx_data_usage.id\n")
            f.write("_pdbx_data_usage.type\n")
            f.write("_pdbx_data_usage.url\n")
            f.write(";NON-COMMERCIAL USE ONLY, BY USING THIS FILE YOU AGREE TO THE TERMS OF USE FOUND\n")
            f.write("AT https://alphafoldserver.com/output-terms\n")
            f.write("; 1 \"Terms of use\" https://alphafoldserver.com/output-terms \n")
            f.write("#\n")
            
            # code details (here "Converted from PDB")
            f.write("_software.classification other\n")
            f.write("_software.date           ?\n")
            f.write("_software.description    \"Structure prediction\"\n")
            f.write("_software.name           AlphaFold\n")
            f.write("_software.pdbx_ordinal   1\n")
            f.write("_software.type           package\n")
            f.write("_software.version        \"Converted from PDB\"\n")
            f.write("#\n")
            
            # maps entity ID to chain ID in the file
            f.write("loop_\n")
            f.write("_struct_asym.entity_id\n")
            f.write("_struct_asym.id\n")
            for chain_id in sorted(chains_info.keys()):
                entity_id = chains_info[chain_id]['entity_id']
                f.write(f"{entity_id} {chain_id} \n")
            f.write("#\n")

            # The middle section: sequence scheme 
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

            temp_cif = cif_file + ".temp"
            io_cif = PDB.MMCIFIO()
            io_cif.set_structure(structure)
            io_cif.save(temp_cif)

            # Ensure consistent naming of chains
            with open(temp_cif, 'r') as temp_f:
                temp_content = temp_f.read()
            
            if '_atom_site.' in temp_content:
                atom_section_start = temp_content.find('loop_\n_atom_site.')
                if atom_section_start != -1:
                    atom_section = temp_content[atom_section_start:]
                    
                    biopython_chains = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                    original_chains = sorted(chains_info.keys())
                    
                    atom_lines = atom_section.split('\n')
                    
                    for line_idx, line in enumerate(atom_lines):
                        if line.startswith('ATOM') or line.startswith('HETATM'):

                            fields = line.split()
                            if len(fields) >= 18:  
                                for i, orig_chain in enumerate(original_chains):
                                    if i < len(biopython_chains) and fields[6] == biopython_chains[i]:
                                        fields[6] = orig_chain
                                        break

                                atom_lines[line_idx] = ' '.join(fields)
                    
                    atom_section = '\n'.join(atom_lines)
                    
                    f.write(atom_section)
            
            # Delete temp file
            if os.path.exists(temp_cif):
                os.remove(temp_cif)
        
        print(f"{pdb_file} converted to {cif_file}")
