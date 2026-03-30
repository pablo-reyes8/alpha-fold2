import json
from pathlib import Path
import pandas as pd

json_path = Path("/content/af_subset/jsons/fb_protein.json")
msa_root = Path("/content/af_subset/foldbench_msas")
cif_root = Path("/content/af_subset/reference_structures")

with open(json_path, "r") as f:
    data = json.load(f)

rows = []

for qname, q in data["queries"].items():
    chain = q["chains"][0]
    chain_ids = chain["chain_ids"]
    sequence = chain["sequence"]

    # elegimos la primera cadena alfabética simple, como hicimos al descargar MSAs
    chosen_chain = None
    for cid in chain_ids:
        if isinstance(cid, str) and len(cid) == 1 and cid.isalpha():
            chosen_chain = cid
            break

    if chosen_chain is None:
        continue

    msa_dir_name = f"{qname.lower()}_{chosen_chain}"
    msa_dir = msa_root / msa_dir_name

    cif_candidates = list(cif_root.glob(f"{qname.lower()}-assembly1_*.cif"))
    cif_file = cif_candidates[0] if len(cif_candidates) > 0 else None

    rows.append({
        "query_name": qname,
        "chain_id": chosen_chain,
        "msa_dir_name": msa_dir_name,
        "msa_exists": msa_dir.exists(),
        "msa_dir": str(msa_dir),
        "cif_exists": cif_file is not None,
        "cif_file": str(cif_file) if cif_file is not None else None,
        "seq_len": len(sequence),
        "sequence": sequence})

df_map = pd.DataFrame(rows)
df_map = df_map.dropna(subset=['cif_file'])
print("\nN total:", len(df_map))
print("MSA disponibles:", df_map["msa_exists"].sum())
print("CIF disponibles:", df_map["cif_exists"].sum())
print("Ambos disponibles:", (df_map["msa_exists"] & df_map["cif_exists"]).sum())

json_path = Path("/content/af_subset/jsons/fb_protein.json")

with open(json_path, "r") as f:
    data = json.load(f)

q = data["queries"]["7QRJ"]

print("query_name:", q["query_name"])
print("chain_ids:", q["chains"][0]["chain_ids"])
print("sequence:")
print(q["chains"][0]["sequence"])
print("length:", len(q["chains"][0]["sequence"]))

msa_path = Path("/content/af_subset/foldbench_msas/7qrj_A/cfdb_hits.a3m")

seqs = []
names = []



