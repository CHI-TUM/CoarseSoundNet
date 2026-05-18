import os


root = "/path/to/trained/models"

exp_folders = os.listdir(root)
print(exp_folders)

for exp_f in exp_folders:
    print("\n")
    exp_dirpath = os.path.join(root, exp_f)
    if not os.path.isdir(exp_dirpath):
        continue
    print(exp_dirpath)

    dirnames = os.listdir(exp_dirpath)
    print("Dirnames: ", dirnames)

    for dn in dirnames:
        if not dn.lower().startswith("epoch_"):
            continue
        
        dir_path = os.path.join(exp_dirpath, dn)
        if not os.path.isdir(dir_path):
            continue

        for file in os.listdir(dir_path):
            if file.endswith(".pt"):
                filepath = os.path.join(dir_path, file)
                os.remove(filepath)
                print("Deleted: ", filepath)