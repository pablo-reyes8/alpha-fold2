
import json
from pathlib import Path
import pandas as pd

json_path = Path("/content/af_subset/jsons/fb_protein.json")

with open(json_path, "r") as f:
    data = json.load(f)

q = data["queries"]["7QRJ"]

print("query_name:", q["query_name"])
print("chain_ids:", q["chains"][0]["chain_ids"])
print("sequence:")
print(q["chains"][0]["sequence"])
print("length:", len(q["chains"][0]["sequence"]))


msa_path = "/content/af_subset/foldbench_msas/7qrj_A/cfdb_hits.a3m"

with open(msa_path, "r") as f:
    lines = f.readlines()

print("Primeras 20 líneas:\n")
for line in lines[:20]:
    print(line.rstrip())



msa_path = Path("/content/af_subset/foldbench_msas/7qrj_A/cfdb_hits.a3m")

seqs = []
names = []

with open(msa_path, "r") as f:
    current_name = None
    current_seq = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_name is not None:
                names.append(current_name)
                seqs.append("".join(current_seq))
            current_name = line[1:]
            current_seq = []
        else:
            current_seq.append(line)
    if current_name is not None:
        names.append(current_name)
        seqs.append("".join(current_seq))

print("Número de secuencias en este archivo:", len(seqs))
print("Longitud de la primera secuencia:", len(seqs[0]))
print("\nPrimeros 5 nombres:")
for n in names[:5]:
    print(n)

print("\nPrimeras 5 secuencias recortadas:")
for s in seqs[:5]:
    print(s[:120])


import py3Dmol

cif_path = "/content/af_subset/reference_structures/7qrj-assembly1_68.cif"

with open(cif_path, "r") as f:
    cif_str = f.read()

view = py3Dmol.view(width=800, height=600)
view.addModel(cif_str, "mmcif")
view.setStyle({"cartoon": {"color": "spectrum"}})
view.zoomTo()
view.show()

import numpy as np

from Bio.PDB.MMCIFParser import MMCIFParser

parser = MMCIFParser(QUIET=True)
structure = parser.get_structure("7qrj", "/content/af_subset/reference_structures/7qrj-assembly1_68.cif")

for model in structure:
    print("Model:", model.id)
    for chain in model:
        print("Chain:", chain.id)
        residues = list(chain.get_residues())
        print("N residuos:", len(residues))
        break
    break

coords = []
res_ids = []

for model in structure:
    for chain in model:
        if chain.id == "A":
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].coord)
                    res_ids.append(residue.id)
            break
    break

coords = np.array(coords)

print("Shape coords:", coords.shape)
print("Primeras 5 coordenadas:")
print(coords[:10])


import numpy as np
import matplotlib.pyplot as plt

D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

plt.figure(figsize=(6,6))
plt.imshow(D)
plt.colorbar(label="Distance (Å)")
plt.title("Cα distance map - 7QRJ chain A")
plt.xlabel("Residue index")
plt.ylabel("Residue index")
plt.show()