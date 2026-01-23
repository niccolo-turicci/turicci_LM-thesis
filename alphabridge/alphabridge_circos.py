# load packages
import json
from pycirclize import Circos
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
from itertools import combinations

# read alphabridge output from input folder
def read_alphabridge_output(path):
    if not path.endswith("/"):
        path += "/"
        
    with open(path+"alphabridge_data.json") as f:
        alphabridge = json.load(f)
    
    with open(path+"network_data.json") as f:
        network = json.load(f)
    return alphabridge, network

# color like alphafold3 based on plddt
def color_residue(value):
    if value < 50:
        return "orange"
    elif 50 <= value < 70:
        return "yellow"
    elif 70 <= value < 90:
        return "lightblue"
    else:
        return "darkblue"

# rename links
def rename_links(links):
    if "-" in links:
        A1 = int(links.split("-")[0])
        A2 = int(links.split("-")[1])+1
    else:
        A1 = int(links)
        A2 = A1 + 1
    return A1, A2


def get_chain_info_from_json(alphabridge, network): 
    name = alphabridge["structure"][0]["job_id"].split("_")
    chains_info = {}
    for s in alphabridge["structure"][0]["chains"]["polymer"]:
        seq = s["sequence"]
        length = len(seq)
        residues = [color_residue(x['plddt']) for x in s["residues"]]
        chains_info[s["label_asym_id"]] = [seq, length, residues]
        
    try:
        nodes = network['network at 0.5']["network_not_merged"]["nodes"]
        links = network['network at 0.5']["network_not_merged"]["links"] # [i]["label"].replace(")","").replace("(","").replace("np.int64","").split()[1:]
        selected_nodes = {}
        selected_links = [] 
        for node in nodes:
            node_id = node["id"]
            node_chain = node["label_asym_id"]
            node_label = node["label"].replace(")","").replace("(","").replace("np.int64","").split()[1:]
            selected_node = (node_chain, node_label)
            if node_id in selected_nodes.keys():
                selected_nodes[node_id].append(selected_node)
            else:
                selected_nodes[node_id] = [selected_node]
        for link in links:
            if link["color"] == "black":
                selected_links.append((link["source"], link["target"]))        
    except KeyError:
        selected_nodes = {}
        selected_links = []
    return name, chains_info, selected_nodes, selected_links

# load json files
def alphabridge_circos(path):    
    alphabridge, network = read_alphabridge_output(path)
    
    # variables definition
    chains_info = get_chain_info_from_json(alphabridge, network)
    iptm = float(alphabridge["structure"][0]["iptm"])
    contact = float(alphabridge["structure"][0]["contact_iptm"])
    palette_links = sns.color_palette(palette='Pastel1')
    palette_sectors = sns.color_palette(palette='Pastel2')
    
    # links definition
    links = []
    for link_info in chains_info[3]:
        if len(link_info) != 0:
            l1, l2 = link_info
            c1 = chains_info[2][l1][0][0]
            c2 = chains_info[2][l2][0][0]
            for link1 in chains_info[2][l1][0][1]:
                l1_start, l1_stop = rename_links(link1)
                for link2 in chains_info[2][l2][0][1]:
                    l2_start, l2_stop = rename_links(link2)
                    links.append([(c1, l1_start, l1_stop), (c2, l2_start, l2_stop)])
    
    # define circos
    sectors = {x:chains_info[1][x][1] for x in network["label_asym_ids"]}
    
    circos = Circos(sectors, space=2)
    circos.text(f"Dimer composed by {chains_info[0][0]} and {chains_info[0][1]}\n\niptm = {iptm:.2f}\n\ncontact iptm = {contact:.2f}", size=13)
    
    color_data = dict(zip(network["label_asym_ids"], palette_sectors))
    residue_data = {x:chains_info[1][x][2] for x in network["label_asym_ids"]}
    
    for sector in circos.sectors:
        sector.text(f"{sector.name}", r=90, size=11)
        outer_track = sector.add_track((75, 80))
        outer_track.axis(fc=color_data[sector.name])
        outer_track.xticks_by_interval(interval=50, label_orientation="horizontal")
        
        rect_track = sector.add_track((70, 75))
        rect_size = 1
        for i in range(int(rect_track.size / rect_size)):
            x1, x2 = i * rect_size, i * rect_size + rect_size
            rect_track.rect(x1, x2, lw=0.5, color=residue_data[sector.name][i])
    pairs = list(combinations(network["label_asym_ids"], 2))
    pairs_colors = dict(zip(pairs, palette_links))
    if len(links) != 0:
        for link in links:
            circos.link(link[0], link[1], color=pairs_colors[(link[0][0], link[1][0])])
    circos.savefig(path+"circos_plot.svg")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python alphabridge_circos.py path/to/alphabridge-output")
        sys.exit(1)

    if not ("alphabridge_data.json" in os.listdir(sys.argv[1]) or "network_data.json" in os.listdir(sys.argv[1])):
        print(f"Folder {sys.argv[1]} does not have alphabridge_data.json and/or network_data.json")
        sys.exit(1)
    
    alphabridge_circos(sys.argv[1])

