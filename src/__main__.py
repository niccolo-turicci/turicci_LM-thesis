import os
import argparse
from .conversion import *

def main():
    parser = argparse.ArgumentParser(description="Takes APD output and makes it suited to be fed into ABridge.")

    # arguments after "python -m src"
    parser.add_argument('--input', type=str, required=True, help="Path to the input folder containing the APD output.")
    parser.add_argument('--output', type=str, required=True, help="Name for output directory (both temporary and output folders are gonna be here.")
    
    args = parser.parse_args()


    # create temporary folder with intermediate files and output folder
    temp_folder = os.path.join(os.getcwd(), args.output, "temp_files")
    output_folder = os.path.join(os.getcwd(), args.output, "output_folder")
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)


    # here are the functions: defined in conversion.py
    output_residue_distances(args.input, temp_folder)
    output_contact_probabilities(args.input, temp_folder)
    output_full_data(args.input, temp_folder, output_folder)
    output_summary_confidence(args.input, output_folder)
    output_job_request(args.input, output_folder)
    convert_pdb_to_cif(args.input, output_folder)

# actual main
if __name__ == "__main__":
    main()
