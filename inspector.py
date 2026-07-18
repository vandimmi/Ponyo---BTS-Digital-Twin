import os
import sys

try:
    sys.path.append("gaussian-splatting")
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary
except ImportError:
    print("Cannot import colmap_loader from gaussian-splatting")

def print_tree(directory, prefix="", max_files=3):
    try:
        entries = os.listdir(directory)
    except PermissionError:
        print(prefix + " [Permission Denied]")
        return
    except FileNotFoundError:
        print(prefix + " [Not Found]")
        return

    entries = [e for e in entries if e != "__MACOSX" and not e.startswith("._")]
    files = [e for e in entries if os.path.isfile(os.path.join(directory, e))]
    dirs = [e for e in entries if os.path.isdir(os.path.join(directory, e))]
    
    shown_entries = dirs + files[:max_files]
    if len(files) > max_files:
        shown_entries.append(f"... and {len(files) - max_files} more files")

    for i, entry in enumerate(shown_entries):
        path = os.path.join(directory, entry) if not entry.startswith("...") else None
        is_last = (i == len(shown_entries) - 1)
        connector = "\\-- " if is_last else "|-- "
        print(prefix + connector + entry)
        
        if path and os.path.isdir(path):
            new_prefix = prefix + ("    " if is_last else "|   ")
            print_tree(path, new_prefix, max_files)

def inspect_bin(path):
    print(f"Inspecting bin file: {path}")
    name = os.path.basename(path)
    try:
        if name == "cameras.bin":
            cams = read_intrinsics_binary(path)
            print(f"  Loaded {len(cams)} cameras.")
        elif name == "images.bin":
            imgs = read_extrinsics_binary(path)
            print(f"  Loaded {len(imgs)} images.")
        elif name == "points3D.bin":
            pts = read_points3D_binary(path)
            print(f"  Loaded {len(pts)} 3D points.")
        else:
            print(f"  Unknown .bin file: {name}")
    except Exception as e:
        print(f"  Error loading {name}: {e}")

def inspect_csv(path):
    print(f"Inspecting csv/txt file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i in range(5):
                line = f.readline()
                if not line:
                    break
                print(f"  Row {i}: {line.strip()}")
    except Exception as e:
        print(f"  Error reading {path}: {e}")

def find_and_inspect(directory):
    for root, dirs, files in os.walk(directory):
        if "__MACOSX" in root:
            continue
        for f in files:
            if f.startswith("._"):
                continue
            path = os.path.join(root, f)
            if f.endswith('.bin'):
                inspect_bin(path)
            elif f.endswith('.csv') or f.endswith('.txt'):
                inspect_csv(path)

def main():
    base_dir = r"d:\Ponyo---BTS-Digital-Twin"
    for fd in ["data", "data_v"]:
        folder = os.path.join(base_dir, fd)
        print(f"\n--- Inspecting structure of {folder} ---")
        if not os.path.exists(folder):
            print(f"{folder} does not exist.")
            continue
        print_tree(folder, max_files=2)
        print("\n--- Reading files ---")
        find_and_inspect(folder)

if __name__ == "__main__":
    main()
