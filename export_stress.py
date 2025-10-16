# save as export_stress_with_coords.py
# run with: abaqus python export_stress_with_coords.py

import os, csv
from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL, NODAL, INTEGRATION_POINT

# --- CONFIG ---
folder = r"C:\Users\lfeltner\Dropbox\SurfaceStressML\StableINPFiles"
outdir = os.path.join(folder, "csv_export")
if not os.path.exists(outdir):
    os.makedirs(outdir)

def last_step_and_frame(odb):
    # steps is an OrderedDict in practice, but be explicit:
    step_names = list(odb.steps.keys())
    step = odb.steps[step_names[-1]]
    frame = step.frames[-1]
    return step, frame

for fname in os.listdir(folder):
    if not fname.lower().endswith(".odb"):
        continue
    odb_path = os.path.join(folder, fname)
    print("Opening:", odb_path)
    odb = openOdb(odb_path, readOnly=True)

    _, frame = last_step_and_frame(odb)

    # Try to get stresses at element-nodal first (common case)
    S = frame.fieldOutputs['S']
    try:
        S_elem_nodal = S.getSubset(position=ELEMENT_NODAL)
        values = S_elem_nodal.values
        pos_used = "ELEMENT_NODAL"
    except Exception:
        values = []
        pos_used = None

    # Fallbacks if needed
    if not values:
        try:
            S_nodal = S.getSubset(position=NODAL)
            values = S_nodal.values
            pos_used = "NODAL"
        except Exception:
            pass
    if not values:
        # As a last resort, use integration point values (no coordinates match).
        # You can skip writing if this happens, or keep as element/gauss point data.
        S_ip = S.getSubset(position=INTEGRATION_POINT)
        values = S_ip.values
        pos_used = "INTEGRATION_POINT"

    print("Stress position used:", pos_used, "(count=%d)" % len(values))

    rows = []
    if pos_used in ("ELEMENT_NODAL", "NODAL"):
        # Build per-instance coordinate maps
        coord_maps = {}
        for instName, inst in odb.rootAssembly.instances.items():
            coord_maps[instName] = {n.label: n.coordinates for n in inst.nodes}

        # Accumulate per unique node (instance,nodeLabel) -> sum/count for averaging
        acc = {}  # (instName, nodeLabel) -> [sum_s11, sum_s22, count]
        for v in values:
            instName = v.instance.name
            nodeLabel = v.nodeLabel
            s11 = v.data[0]
            s22 = v.data[1]
            key = (instName, nodeLabel)
            if key not in acc:
                acc[key] = [0.0, 0.0, 0]
            acc[key][0] += s11
            acc[key][1] += s22
            acc[key][2] += 1

        # Average and attach coordinates
        for (instName, nodeLabel), (sum11, sum22, cnt) in acc.items():
            x, y, z = coord_maps[instName][nodeLabel]
            rows.append((nodeLabel, x, y, z, sum11 / cnt, sum22 / cnt))

    else:
        # INTEGRATION_POINT fallback: no single node coordinate; use element centroid approx
        # or skip writing. Here we skip to avoid misleading coordinates.
        print("No nodal/elements-nodal stresses available; skipping nodal CSV for", fname)
        odb.close()
        continue

    # Sort rows by node for stable output
    rows.sort(key=lambda r: r[0])

    # Write CSV (Py2 on Windows => 'wb')
    csv_name = os.path.splitext(fname)[0] + "_coords_S11_S22.csv"
    csv_path = os.path.join(outdir, csv_name)
    with open(csv_path, "wb") as f:
        w = csv.writer(f)
        w.writerow(["Node", "X", "Y", "Z", "S11", "S22"])
        w.writerows(rows)

    odb.close()
    print("Saved:", csv_path, "rows:", len(rows))

print("Done.")
