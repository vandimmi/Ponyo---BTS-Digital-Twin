import nbformat as nbf
import os

def append_files_to_notebook():
    nb_path = 'kaggle_notebook.ipynb'
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)

    # We want to insert the file creation cells before the training cell (cell index 6 currently)
    # Let's just create them as a list of cells and insert them.
    files_to_inject = [
        "gaussian-splatting/render_test_poses.py",
        "gaussian-splatting/test_pose_loader.py",
        "gaussian-splatting/package_submission.py",
        "gaussian-splatting/eval.py",
        "run_all_scenes.py",
        "verify_submission.py"
    ]
    
    new_cells = []
    for filepath in files_to_inject:
        if os.path.exists(filepath):
            with open(filepath, 'r') as file_in:
                content = file_in.read()
            # %%writefile writes to the current directory, which is /kaggle/working/repo if we chdir there
            # Since some files belong in gaussian-splatting, let's specify the path correctly relative to repo
            # Wait, the repo IS gaussian-splatting for them?
            # "The user's repo is a fork/clone of gaussian-splatting"
            # If their repo IS gaussian-splatting, then we just write to the root of the repo.
            basename = os.path.basename(filepath)
            cell_content = f"%%writefile {basename}\n" + content
            cell = nbf.v4.new_code_cell(cell_content)
            new_cells.append(cell)
            print(f"Added {basename} to notebook.")
    
    # We also need to apply the train.py modifications!
    # Instead of doing it via python replace, let's just write the patched train.py entirely.
    if os.path.exists("gaussian-splatting/train.py"):
        with open("gaussian-splatting/train.py", 'r') as file_in:
            train_content = file_in.read()
        cell_content = "%%writefile train.py\n" + train_content
        new_cells.append(nbf.v4.new_code_cell(cell_content))
        print("Added patched train.py to notebook.")

    # Insert after cell 4 (the pip install cell)
    # Let's find cell 4 (the one that prints "Installed CUDA extensions successfully!")
    insert_idx = 4
    for i, cell in enumerate(nb.cells):
        if "Installed CUDA extensions successfully" in cell.source:
            insert_idx = i + 1
            break
            
    nb.cells = nb.cells[:insert_idx] + new_cells + nb.cells[insert_idx:]

    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
    
    print("Notebook updated successfully.")

if __name__ == '__main__':
    append_files_to_notebook()
