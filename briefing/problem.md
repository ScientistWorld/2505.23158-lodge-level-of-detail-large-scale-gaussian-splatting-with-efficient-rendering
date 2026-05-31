# Problem Definition

## Research Question

How can a large 3D Gaussian Splatting scene be rendered from novel viewpoints with high image quality while reducing the number of Gaussian primitives that must be processed and resident in memory?

## Why It Matters

Large scenes require many fine primitives. Rendering all of them for every camera wastes computation on distant detail and can exceed the memory budget of mobile or low-power devices.

## Success Criteria

A successful system should render held-out views close to the reference images, keep structural similarity high, and substantially reduce both visible and loaded Gaussian counts relative to a full-scene renderer. These counts are the benchmark's hardware-independent efficiency constraints.
