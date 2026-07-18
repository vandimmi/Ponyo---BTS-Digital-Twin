import re
import sys

def modify_train_py():
    train_py_path = r'd:\Ponyo---BTS-Digital-Twin\gaussian-splatting\train.py'
    with open(train_py_path, 'r') as f:
        content = f.read()

    # 1. Add args for auto-zip and auto-save dir to parser
    if "--kaggle_working_dir" not in content:
        content = content.replace(
            "parser.add_argument(\"--start_checkpoint\", type=str, default = None)",
            "parser.add_argument(\"--start_checkpoint\", type=str, default = None)\n    parser.add_argument(\"--kaggle_working_dir\", type=str, default='/kaggle/working/')\n    parser.add_argument(\"--resume\", action='store_true', help='Auto-resume from latest checkpoint in model_path')"
        )

    # 2. Add checkpoint loading logic to resume properly
    # In 3DGS, `torch.load(checkpoint)` is done in train.py. However, the optimizer state is not saved in `gaussians.capture()`.
    # Wait, in gaussian_model.py, capture() returns: 
    # (self.active_sh_degree, self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation, self._opacity, self.max_radii2D, self.xyz_gradient_accum, self.denom, self.optimizer.state_dict(), self.spatial_lr_scale)
    # Ah! It DOES save the optimizer state_dict()!
    # And restore() does `self.optimizer.load_state_dict(model_args[10])`!
    # Let's check capture in gaussian_model.py. Actually, we don't need to check, 3DGS natively saves it in capture().
    
    resume_logic = """
    # Resume logic
    if args.resume and not args.start_checkpoint:
        import glob
        chkpts = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
        if len(chkpts) > 0:
            # Sort by iteration number
            chkpts.sort(key=lambda x: int(re.findall(r"chkpnt(\\d+).pth", x)[0]))
            args.start_checkpoint = chkpts[-1]
            print(f"Resuming from {args.start_checkpoint}")
"""
    if "if args.resume" not in content:
        content = content.replace(
            "print(\"Optimizing \" + args.model_path)",
            resume_logic + "\n    print(\"Optimizing \" + args.model_path)"
        )

    # 3. Add memory printing and data_device 
    # 3DGS DatasetParams usually has data_device. Let's add VRAM printing in the training loop
    vram_print = """
        if iteration % 1000 == 0:
            print(f"\\n[ITER {iteration}] Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
"""
    if "Peak VRAM" not in content:
        content = content.replace(
            "gaussians.update_learning_rate(iteration)",
            "gaussians.update_learning_rate(iteration)" + vram_print
        )

    # 4. Modify checkpoint save path to also copy to /kaggle/working/
    save_logic = """
            if (iteration in checkpoint_iterations):
                print("\\n[ITER {}] Saving Checkpoint".format(iteration))
                save_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                torch.save((gaussians.capture(), iteration), save_path)
                import shutil
                if os.path.exists(opt.kaggle_working_dir):
                    shutil.copy(save_path, opt.kaggle_working_dir)
"""
    
    # Let's replace the native save block
    content = re.sub(
        r"if \(iteration in checkpoint_iterations\):.*?torch\.save\(\(gaussians\.capture\(\), iteration\), scene\.model_path \+ \"/chkpnt\" \+ str\(iteration\) \+ \".pth\"\)",
        save_logic.strip(),
        content,
        flags=re.DOTALL
    )

    # 5. Zip at the end of the run
    zip_logic = """
    # All done
    print("\\nTraining complete.")
    if os.path.exists(args.kaggle_working_dir):
        import zipfile
        out_zip = os.path.join(args.kaggle_working_dir, "training_run.zip")
        print(f"Zipping outputs to {out_zip}")
        with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(args.model_path):
                for f in files:
                    # Zip everything in the output directory
                    abs_path = os.path.join(root, f)
                    rel_path = os.path.relpath(abs_path, args.model_path)
                    zipf.write(abs_path, rel_path)
"""
    if "Zipping outputs" not in content:
        content = content.replace(
            "# All done\n    print(\"\\nTraining complete.\")",
            zip_logic.strip()
        )
        
    # We must pass args to training() to make kaggle_working_dir accessible if we use it, 
    # but wait, `opt` is passed to training(). Let's just use `os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working/")` inside `training` instead.
    
    content = content.replace("opt.kaggle_working_dir", "os.environ.get('KAGGLE_WORKING_DIR', '/kaggle/working/')")

    with open(train_py_path, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    modify_train_py()
