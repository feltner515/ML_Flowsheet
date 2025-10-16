import pandas as pd
import numpy as np
import os
from tqdm import tqdm

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

def _match_z_by_xy(data_df, pos_df, xy_tol=0.02):
    """Attach Z to each (x,y) in data_df by nearest (X,Y) in pos_df."""
    if KDTree is None:
        pos_xy = pos_df[['X','Y']].to_numpy()
        pos_Z  = pos_df['Z'].to_numpy()
        zs = []
        for x, y in data_df[['x','y']].to_numpy():
            d2 = (pos_xy[:,0]-x)**2 + (pos_xy[:,1]-y)**2
            j  = int(np.argmin(d2))
            if xy_tol is not None and np.sqrt(d2[j]) > xy_tol:
                raise ValueError(f"No position within {xy_tol} of ({x:.3f},{y:.3f}). Min dist {np.sqrt(d2[j]):.4f}")
            zs.append(pos_Z[j])
        out = data_df.copy()
        out['Z'] = zs
        return out
    else:
        tree = KDTree(pos_df[['X','Y']].to_numpy())
        d, idx = tree.query(data_df[['x','y']].to_numpy(), k=1)
        if xy_tol is not None and np.any(d > xy_tol):
            bad = np.where(d > xy_tol)[0][:5]
            ex  = ", ".join([f"i={i}, d={d[i]:.4f}"] for i in bad)
            raise ValueError(f"{(d>xy_tol).sum()} impacts exceed xy_tol={xy_tol}. Examples: {ex}")
        out = data_df.copy()
        out['Z'] = pos_df['Z'].to_numpy()[idx]
        return out

def create_full_resolution_energy_stack(position_file,
                                        data_file,
                                        velocity,
                                        grid_size=(251, 251),
                                        domain_size=(5.0, 5.0),
                                        xy_tol=0.02,
                                        z_order='asc'):
    """
    Build energy stack E[x,y,t] ordered by Z.
    Spatial kernel: uniform DISK of radius = impactdiameter/2.
    Disk normalized by area so sum over pixels ≈ total kinetic energy E_k.
    """
    print(f"\n--- Processing {os.path.basename(data_file)} ---")
    pos_df  = pd.read_csv(position_file)
    data_df = pd.read_csv(data_file)

    # Attach Z
    df = _match_z_by_xy(data_df, pos_df, xy_tol=xy_tol)
    ascending = (z_order.lower() == 'asc')
    df = df.sort_values('Z', ascending=ascending).reset_index(drop=True)

    print(f" Impacts: {len(df)} (Z: {df['Z'].min()} → {df['Z'].max()})")
    print(f" Grid: {grid_size}, Domain: {domain_size} (units same as x,y)")

    # Grid
    nx, ny = grid_size
    Lx, Ly = domain_size
    x_edges = np.linspace(0, Lx, nx + 1)
    y_edges = np.linspace(0, Ly, ny + 1)
    x_centers = 0.5*(x_edges[:-1] + x_edges[1:])
    y_centers = 0.5*(y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='ij')
    pixel_area = (Lx / nx) * (Ly / ny)

    T = len(df)
    energy_stack = np.zeros((nx, ny, T), dtype=np.float32)

    rho = 7800.0  # kg/m^3 (steel)

    for i, row in df.iterrows():
        x0 = float(row['x'])
        y0 = float(row['y'])
        R  = float(row['impactdiameter']) / 2.0

        # Kinetic energy
        d_eq_m = float(row['AreaEqDiameter']) * 1e-6
        vol_m3 = (np.pi / 6.0) * (d_eq_m**3)
        mass   = rho * vol_m3
        E_k    = 0.5 * mass * (velocity**2)

        # Disk
        disk_area = np.pi * (R**2)
        if disk_area <= 0:
            continue
        energy_density = E_k / disk_area
        r2 = (Xc - x0)**2 + (Yc - y0)**2
        mask = (r2 <= R*R)
        deposit = np.zeros((nx, ny), dtype=np.float32)
        deposit[mask] = (energy_density * pixel_area)

        energy_stack[..., i] = deposit

        if (i+1) % 50 == 0 or (i+1) == T:
            print(f"   Impact {i+1}/{T}: center=({x0:.2f},{y0:.2f}), R={R:.3f}, E={E_k:.2e} J")

    return energy_stack

def process_directory_for_energy_stacks(directory,
                                        velocity=65.0,
                                        grid_size=(251, 251),
                                        domain_size=(5.0, 5.0),
                                        xy_tol=0.02,
                                        z_order='asc'):
    files = [f for f in os.listdir(directory) if f.endswith("_shot_positions.csv")]
    for file in tqdm(files, desc="Files"):
        base = file.replace("_shot_positions.csv", "")
        pos_path  = os.path.join(directory, file)
        data_path = os.path.join(directory, base + ".txt")
        if not os.path.exists(data_path):
            print(f" ⚠️ Skipping {base}: missing {base}.txt")
            continue
        stack = create_full_resolution_energy_stack(
            pos_path, data_path, velocity,
            grid_size=grid_size, domain_size=domain_size,
            xy_tol=xy_tol, z_order=z_order
        )
        out_path = os.path.join(directory, base + "_energystack.npy")
        np.save(out_path, stack)
        print(f" ✅ Saved {out_path} with shape {stack.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate full-resolution energy stacks (disk kernel) ordered by Z.")
    parser.add_argument("directory", type=str, help="Directory with *_shot_positions.csv and corresponding .txt files")
    parser.add_argument("--velocity", type=float, default=65.0, help="Impact velocity in m/s")
    parser.add_argument("--xy_tol", type=float, default=0.02, help="Max XY mismatch when pairing (same units as x,y)")
    parser.add_argument("--z_order", type=str, default="asc", choices=["asc","desc"], help="Sort by Z ascending or descending")
    args = parser.parse_args()
    process_directory_for_energy_stacks(args.directory, velocity=args.velocity,
                                        xy_tol=args.xy_tol, z_order=args.z_order)