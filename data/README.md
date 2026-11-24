# data/

- `gsrender.py`: the public renderer and the representation protocol. It defines the submission format, how resident sets and blending are chosen per camera, and how loaded and visible Gaussians are counted. The hidden evaluator uses an identical copy.
- `lodge_city/base_gaussians.npz`: the trained full 3DGS reconstruction (247,095 Gaussians). Arrays: `means`, `scales` (std devs in metres), `quats` (w,x,y,z), `opacities`, `colors` (RGB).
- `lodge_city/cameras.json`: intrinsics (f = 96, 160×96 px) and the camera sets `train`, `val_on_path`, `val_flythrough` and `val_off_path`. Each camera is given as eye, yaw and pitch, in metres and radians; z is up.
- `lodge_city/images.npz`: uint8 target images for each public camera set, keyed by set name.

The data is synthetic and agent-generated. It was produced once by the hidden world builder and committed here.
