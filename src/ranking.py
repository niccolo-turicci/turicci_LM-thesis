import os
import json

# --- 7 - Renames the files based on the iptm score (summary_confidences file) ---

# finds the iptm scores
def get_iptm_ranking(output_folder):
    iptm_list = []
    for i in range(5):
        summary_path = os.path.join(output_folder, f"run_out_summary_confidences_{i}.json")
        with open(summary_path) as f:
            summary = json.load(f)
        iptm = summary.get("iptm", 0)
        iptm_list.append((i, iptm))

    iptm_sorted = sorted(iptm_list, key=lambda x: x[1], reverse=True)
    
    return [old_idx for old_idx, _ in iptm_sorted]

# renumbers files
def renumber_files_by_iptm(output_folder):
    ranked_indices = get_iptm_ranking(output_folder)
    file_types = [
        "run_out_full_data_{}.json",
        "run_out_model_{}.cif",
        "run_out_summary_confidences_{}.json"
       
    ]
    temp_files = []
    
    for i in range(5):  # temporary renaming
        for fpattern in file_types:
            old_file = os.path.join(output_folder, fpattern.format(i))
            if os.path.exists(old_file):
                temp_file = os.path.join(output_folder, fpattern.format(f"temp_{i}"))
                os.rename(old_file, temp_file)
                temp_files.append((fpattern, i, temp_file))

    for new_idx, old_idx in enumerate(ranked_indices):  # renames temporary files from 0 to 4
        for fpattern in file_types:
            temp_file = os.path.join(output_folder, fpattern.format(f"temp_{old_idx}"))
            final_file = os.path.join(output_folder, fpattern.format(new_idx))
            if os.path.exists(temp_file):
                os.rename(temp_file, final_file)
                print(f"{temp_file} renamed into {final_file} based on iptm")
