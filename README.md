# cas-blender-tools
Blender python scripts for rendering

## render_interpolations.py

Load models from input folder and renders them while rotating them by 1 degree

Run:

```bash
blender --background --python render_interpolations.py -- --input_dir models --output_dir output --resolution 1024 --engine CYCLES
```

**input_dir**: The path to load models. Default: models  
**output_dir**: The path the output will be dumped to. Default: output  
**resolution**: Resolution of the images. Default: 1024  
**engine**: Blender internal engine for rendering. E.g. 'CYCLES', 'BLENDER_EEVEE'. Default: 'BLENDER_EEVEE'  

## render_batch.py

Load models from input folder and renders them from different points of view

Run:

```bash
blender --background --python render_batch.py -- --input_dir models --output_dir output --resolution 1024 --engine CYCLES
```

**input_dir**: The path to load models. Default: models  
**output_dir**: The path the output will be dumped to. Default: output  
**resolution**: Resolution of the images. Default: 1024  
**engine**: Blender internal engine for rendering. E.g. 'CYCLES', 'BLENDER_EEVEE'. Default: 'BLENDER_EEVEE'  
  
Based on [[https://github.com/panmari/stanford-shapenet-renderer/tree/master]]
