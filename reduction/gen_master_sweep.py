#!/usr/bin/env python
"""Generate the COMPLETE-MASTER-BASIS sweep: one standard Kira job (symmetries
on, unpatched behavior) per canonical non-trivial sector, at TWO seed depths
(stability check: the master list must be identical at both). Each job:
  sectors: [S], r = L+2/s = 2 (depth A) and r = L+3/s = 3 (depth B),
  corner as mandatory target, initiate+triangular only (masters emerge there;
  no back-substitution needed), export via the masters file.
Writes results/gr_master_sweep/S<sec>_<depth>/ + sweep_dirs.txt + a Condor
submit file. Run: condor_submit results/gr_master_sweep/sweep.sub
"""
import os, pickle
ROOT = "/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2"
G = os.path.join(ROOT, "topology_input/gravity3L")
OUT = os.path.join(ROOT, "results/gr_master_sweep")
os.makedirs(OUT, exist_ok=True)

cm = pickle.load(open(os.path.join(ROOT, "results/canonical_sectors_GR_v2.pkl"), "rb"))
canonical = list(cm["canonical"])
triv = set(int(x) for x in open(os.path.join(
    G, "kira_validate/sectormappings/GR/trivialsector")).read().strip().split(","))

N_DEN, N_IND = 10, 15
JOBS = []
n_triv = 0
for S in sorted(canonical):
    if S in triv:
        n_triv += 1
        continue
    L = bin(S).count("1")
    corner = ",".join(["1" if S >> i & 1 else "0" for i in range(N_DEN)]
                      + ["0"] * (N_IND - N_DEN))
    for depth, (r, s) in (("A", (L + 2, 2)), ("B", (L + 3, 3))):
        d = os.path.join(OUT, f"S{S}_{depth}")
        os.makedirs(os.path.join(d, "config"), exist_ok=True)
        for f in ("integralfamilies.yaml", "kinematics.yaml"):
            with open(os.path.join(G, f)) as src, \
                 open(os.path.join(d, "config", f), "w") as dst:
                dst.write(src.read())
        with open(os.path.join(d, "target_integrals"), "w") as f:
            f.write(f"GR[{corner}]\n")
        with open(os.path.join(d, "jobs.yaml"), "w") as f:
            # NO select_integrals: Kira reduces the FULL seeded box, so the
            # masters output is the sector's true slate at these depths.
            # (v1 used select_mandatory_list = corner — target-driven
            # selection, which reports only the corner-subsystem's masters:
            # the S885 job found 1 master where FIRE needs 5. Archived in
            # results/gr_master_sweep_v1_cornerselect.)
            f.write(f"""jobs:
  - reduce_sectors:
      reduce:
        - {{sectors: [{S}], r: {r}, s: {s}}}
      run_initiate: true
      run_triangular: true
      run_back_substitution: false
""")
        JOBS.append(d)

with open(os.path.join(OUT, "sweep_dirs.txt"), "w") as f:
    f.write("\n".join(JOBS) + "\n")

with open(os.path.join(OUT, "sweep_worker.sh"), "w") as f:
    f.write("""#!/bin/bash
# Condor wrapper: run one standard-Kira sweep job in the given directory.
DIR="$1"
export FERMATPATH=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/fire/FIRE7/extra/fuel/extra/ferl64/fer64
cd "$DIR" || exit 1
/het/p4/dshih/jet_images-deep_learning/IBPreduction/kira/kira-3.1 --parallel=2 jobs.yaml > run.log 2>&1
echo "exit $? $(date +%H:%M:%S)" >> run.log
""")
os.chmod(os.path.join(OUT, "sweep_worker.sh"), 0o755)

with open(os.path.join(OUT, "sweep.sub"), "w") as f:
    f.write(f"""universe = vanilla
executable = {OUT}/sweep_worker.sh
arguments = "$(dir)"
request_cpus = 2
request_memory = 6GB
request_disk = 2GB
log = {OUT}/condor.log
output = {OUT}/condor_$(Cluster)_$(Process).out
error = {OUT}/condor_$(Cluster)_$(Process).err
queue dir from {OUT}/sweep_dirs.txt
""")

print(f"canonical sectors: {len(canonical)}; trivial skipped: {n_triv}; "
      f"jobs generated: {len(JOBS)} ({len(JOBS)//2} sectors x 2 depths)")
print(f"submit: condor_submit {OUT}/sweep.sub")
