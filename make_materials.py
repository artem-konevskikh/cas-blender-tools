import os
import json

materials_dir = 'materials'
materials = []

for subdir in os.listdir(materials_dir):
    subdir_path = os.path.join(materials_dir, subdir)
    if os.path.isdir(subdir_path):
        print(subdir_path)
        mat = {
            'name': subdir,
            'albedo': '',
            'normal': '',
            'roughness': '',
            'displacement': ''
        }
        filenames = os.listdir(subdir_path)
        for fn in filenames:
            # print(fn)
            if fn.endswith('.exr'):
                continue
            if 'Albedo' in str(fn):
                mat['albedo'] = os.path.join(materials_dir, subdir, str(fn))
            elif 'Normal' in str(fn):
                mat['normal'] = os.path.join(materials_dir, subdir, str(fn))
            elif 'Roughness' in str(fn):
                mat['roughness'] = os.path.join(materials_dir, subdir, str(fn))
            elif 'Displacement' in str(fn):
                mat['displacement'] = os.path.join(materials_dir, subdir, str(fn))
        materials.append(mat)

with open('materials.json', 'w') as f:
    json.dump(materials, f, indent=4)
