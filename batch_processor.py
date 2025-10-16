import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def per_snapshot_energy_stacks(
    csv_path,
    velocity_m_s=65.0,
    grid_size=(251, 251),          # (nx, ny)
    domain_size_mm=(5.0, 5.0),     # (Lx, Ly) in mm
    steel_density_kg_m3=7800.0,    # base density
    cumulative=False,              # if True: frame t is sum of frames 0..t
    out_dir=None,
    dtype=np.float32,
    save_mode="energy",            # "energy" -> J/pixel, "density" -> J/mm^2
    check_energy=True,             # print mismatch if per-frame sum != E_k
    max_warn=5,                    # cap # of mismatch warnings
):
    """
    Build one 3D energy tensor per snapshot id from a CSV with columns:
      #, mode, X [mm], Y [mm], x_c [µm] (curvature DIAMETER), rho_eff [unitless].

    For each snapshot s, produces E_s of shape (nx, ny, T_s), where T_s is the
    number of impacts in that snapshot. Frame k contains the energy from the
    k-th impact (CSV order) as a uniform disk footprint.

    Units:
      - X,Y in mm (grid in mm).
      - x_c is a DIAMETER in µm -> radius R = x_c/2.
      - Mass/energy computed in SI (m, kg, J).
      - rho_eff multiplies steel density.
      - save_mode="energy": J per pixel; "density": J/mm^2.
    """
    df = pd.read_csv(csv_path)
    required = {"#", "X", "Y", "x_c", "rho_eff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    base = os.path.splitext(os.path.basename(csv_path))[0]
    if out_dir is None:
        out_dir = os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Spatial grid (mm); also precompute pixel area in m^2 and mm^2
    nx, ny       = grid_size
    Lx_mm, Ly_mm = domain_size_mm
    x_edges      = np.linspace(0, Lx_mm, nx + 1)
    y_edges      = np.linspace(0, Ly_mm, ny + 1)
    x_centers    = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers    = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc_mm, Yc_mm = np.meshgrid(x_centers, y_centers, indexing="ij")
    pixel_area_mm2 = (Lx_mm / nx) * (Ly_mm / ny)
    pixel_area_m2  = pixel_area_mm2 * 1e-6  # mm^2 -> m^2

    out_paths = {}
    snapshots = sorted(df["#"].unique(), reverse=True)  # e.g., 100..1
    warn_count = 0

    for s in tqdm(snapshots, desc="Snapshots"):
        d = df[df["#"] == s].copy()  # keep CSV order
        T = len(d)
        if T == 0:
            continue

        # Allocate (nx, ny, T)
        E = np.zeros((nx, ny, T), dtype=dtype)

        # Columns (vectors)
        X_mm    = d["X"].to_numpy(float)        # mm
        Y_mm    = d["Y"].to_numpy(float)        # mm
        xc_um   = d["x_c"].to_numpy(float)      # µm (DIAMETER for footprint & mass)
        rho_eff = d["rho_eff"].to_numpy(float)  # unitless

        # Radius in mm for rasterization; diameter in m for mass
        R_mm   = 0.5 * (xc_um * 1e-3)           # µm -> mm; then /2
        d_m    = xc_um * 1e-6                   # µm -> m (diameter)
        vol_m3 = (np.pi / 6.0) * (d_m ** 3)     # sphere volume by diameter
        m_eff  = rho_eff * steel_density_kg_m3 * vol_m3
        E_imp  = 0.5 * m_eff * (velocity_m_s ** 2)  # J

        for k, (x0, y0, R, Ein) in enumerate(zip(X_mm, Y_mm, R_mm, E_imp)):
            if not np.isfinite(Ein) or not np.isfinite(R) or R <= 0:
                continue

            # Disk mask on mm grid
            r2   = (Xc_mm - x0)**2 + (Yc_mm - y0)**2
            mask = (r2 <= R * R)
            if not np.any(mask):
                continue

            # Energy density (J/m^2) from a uniform disk in SI
            disk_area_m2 = np.pi * (R**2) * 1e-6  # mm^2 -> m^2
            if disk_area_m2 <= 0 or not np.isfinite(disk_area_m2):
                continue
            edens_J_per_m2 = Ein / disk_area_m2   # J/m^2

            if save_mode == "energy":
                # Per-pixel energy (J)
                frame = np.zeros((nx, ny), dtype=dtype)
                frame[mask] = edens_J_per_m2 * pixel_area_m2  # J
                frame_sum = float(frame.sum())
                if check_energy and not np.isclose(frame_sum, Ein, rtol=1e-2, atol=1e-12):
                    if warn_count < max_warn:
                        print(f"⚠ Snapshot {s} impact {k}: sum(frame)={frame_sum:.3e} J vs E={Ein:.3e} J")
                    warn_count += 1
            elif save_mode == "density":
                # Energy density (J/mm^2)
                frame = np.zeros((nx, ny), dtype=dtype)
                frame[mask] = edens_J_per_m2 * 1e-6  # m^-2 -> mm^-2
                # (No sum check—this is density, not energy)
            else:
                raise ValueError("save_mode must be 'energy' or 'density'")

            if cumulative and k > 0:
                E[..., k] = E[..., k-1] + frame
            else:
                E[..., k] = frame

        out_path = os.path.join(out_dir, f"{base}_snap{s:03d}_E.npz")
        np.savez_compressed(out_path, E=E)
        out_paths[int(s)] = {"path": out_path, "shape": tuple(E.shape)}

    if warn_count > max_warn:
        print(f"⚠ Energy-sum warnings suppressed after {max_warn} (total {warn_count}).")
    return out_paths


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Save a 3D energy tensor per snapshot (#) from a CSV.")
    p.add_argument("csv", type=str, help="Path to XXX.csv with columns: #, mode, X, Y, x_c[µm], rho_eff.")
    p.add_argument("--velocity", type=float, default=65.0, help="Impact velocity (m/s).")
    p.add_argument("--nx", type=int, default=251)
    p.add_argument("--ny", type=int, default=251)
    p.add_argument("--Lxmm", type=float, default=5.0)
    p.add_argument("--Lymm", type=float, default=5.0)
    p.add_argument("--rho", type=float, default=7800.0, help="Steel density (kg/m^3).")
    p.add_argument("--cumulative", action="store_true", help="Make frames cumulative over impacts.")
    p.add_argument("--out", type=str, default=None, help="Output folder (defaults to CSV folder).")
    p.add_argument("--savemode", type=str, default="energy", choices=["energy","density"],
                   help="Save J per pixel ('energy') or J/mm^2 ('density').")
    p.add_argument("--no-check", action="store_true", help="Disable per-frame energy sum check.")
    args = p.parse_args()

    info = per_snapshot_energy_stacks(
        args.csv,
        velocity_m_s=args.velocity,
        grid_size=(args.nx, args.ny),
        domain_size_mm=(args.Lxmm, args.Lymm),
        steel_density_kg_m3=args.rho,
        cumulative=args.cumulative,
        out_dir=args.out,
        save_mode=args.savemode,
        check_energy=(not args.no_check),
    )
    for s, d in sorted(info.items()):
        print(f"Snapshot #{s}: saved {d['path']}  shape={d['shape']}")