import nbformat as nbf
import os

def main():
    nb = nbf.v4.new_notebook()

    cell1 = nbf.v4.new_code_cell("""
# 1. Check GPU available, print torch/CUDA version (T4 accelerator assumed enabled).
import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("Device Name:", torch.cuda.get_device_name(0))
""")

    cell2 = nbf.v4.new_code_cell("""
# 2. Set TORCH_CUDA_ARCH_LIST="7.5" (T4 = Turing).
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
print("Set TORCH_CUDA_ARCH_LIST to", os.environ["TORCH_CUDA_ARCH_LIST"])
""")

    cell3 = nbf.v4.new_code_cell("""
# 3. Clone my repo recursively into /kaggle/working/.
# Replace <my-username>/<my-repo> with the actual repository
!git clone --recursive https://github.com/<my-username>/<my-repo>.git /kaggle/working/repo
import os
os.chdir('/kaggle/working/repo')
print("Changed directory to:", os.getcwd())
""")

    cell4 = nbf.v4.new_code_cell("""
# 4. Install requirements.txt, then build diff-gaussian-rasterization and simple-knn from source.
!pip install -q -r requirements.txt
!pip install -q submodules/diff-gaussian-rasterization
!pip install -q submodules/simple-knn
print("Installed CUDA extensions successfully!")
""")

    cell5 = nbf.v4.new_code_cell("""
# 5. Point to dataset at /kaggle/input/<dataset-name>/, print its structure to confirm visibility.
import os
dataset_path = "/kaggle/input/<dataset-name>/"
print(f"Dataset path: {dataset_path}")
if os.path.exists(dataset_path):
    print("Dataset structure:")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files[:3]:
            print('{}{}'.format(subindent, f))
        if len(files) > 3:
            print('{}{}... and {} more files'.format(subindent, len(files)-3))
else:
    print("Dataset directory not found on Kaggle! Please attach the dataset.")
""")

    cell6 = nbf.v4.new_code_cell("""
# 6. Sanity-check import of my repo's loader/model classes (no full training).
import sys
# Make sure we can import from the repo
if '/kaggle/working/repo' not in sys.path:
    sys.path.append('/kaggle/working/repo')

# The user's repo is a fork/clone of gaussian-splatting
from scene.dataset_readers import readColmapSceneInfo
from scene.gaussian_model import GaussianModel
print("Successfully imported readColmapSceneInfo and GaussianModel!")
""")

    cell7 = nbf.v4.new_code_cell("""
# 7. Placeholder cell for training with commented-out args to fill in.
# !python train.py -s /kaggle/input/<dataset-name>/data/HCM0421 --iterations 30000 --eval
print("Uncomment the line above and set the correct dataset path to start training.")
""")

    cell8 = nbf.v4.new_code_cell("""
# 8. Placeholder cell for rendering test poses + zipping submission.zip.
# !python render_test_poses.py -m output/xxxx --test_poses_csv /kaggle/input/<dataset-name>/data/HCM0421/test/test_poses.csv --output_dir renders/
# !python package_submission.py
print("Uncomment the lines above to render test poses and package the submission.zip.")
""")

    nb.cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8]
    
    with open('kaggle_notebook.ipynb', 'w') as f:
        nbf.write(nb, f)
    
    print("Notebook created.")

if __name__ == '__main__':
    main()
