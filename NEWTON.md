## Newton: NVIDIA, DeepMind, and Disney

As of **March 2026**, NVIDIA has officially released **Newton 1.0 (General Availability)**, following an initial open-source beta in September 2025.

Newton is a production-ready, open-source physics engine developed jointly by NVIDIA, Google DeepMind, and Disney Research under the Linux Foundation, with additional collaborations from Toyota Research Institute (TRI).

### Technical Foundation

- **Architecture:** Built on NVIDIA Warp and OpenUSD, providing a modular framework for unified physics simulation.
- **Solvers:** Integrates MuJoCo-Warp, Kamino (Disney), and a new **Vertex Block Descent** solver for high-fidelity deformable materials (cables, cloth, volumetric objects).
- **Advanced Contact:** Features signed distance field (SDF) collision detection and hydroelastic contact modeling for industry-leading precision.
- **Integration:** Seamlessly integrated with **Isaac Lab 3.0**, NVIDIA's open-source robot learning framework.

### GTC 2026 Performance Claims

Running on NVIDIA Blackwell (RTX 6000 Ada) GPUs, Newton 1.0 delivers:

- **252x acceleration** for humanoid locomotion vs MJX.
- **475x acceleration** for in-hand manipulation vs MJX.
- **High-fidelity deformables:** Native support for complex wire/cable handling and soft-body interaction.

### Industry Adoption and Use Cases

- **Samsung:** Utilizing Newton for intricate cable handling and component placement in refrigerator assembly.
- **Skild AI:** Training dexterous manipulation policies with Isaac Lab and Newton.
- **Toyota Research Institute (TRI):** Enhancing solver development and contact modeling for home robotics.
- **Disney Research:** Powering the next generation of Star Wars-inspired BDX droids and Olaf robots with advanced locomotion.

Newton has successfully bridged the gap between MuJoCo's physics transparency and NVIDIA's GPU-native parallelism, becoming the standard for end-to-end robot learning and digital twin deployment.

## Understanding the Core Tech: SDF & Hydroelastic Contact

To achieve its high fidelity, Newton leverages two critical technologies that move beyond traditional physics engines.

### 1. Signed Distance Fields (SDF)

Traditional engines often simplify complex robots into basic shapes (spheres, boxes) or use "point-on-mesh" collision, which is computationally expensive and prone to errors. **SDF** represents geometry as a volumetric field where every point in space stores its distance to the nearest surface:
- **Positive values:** Outside the object.
- **Negative values:** Inside the object.
- **Zero:** Exactly on the surface.

**Why it matters:** SDFs allow for pixel-perfect collision detection on complex parts (like gears or hands) and provide "gradients"—mathematically defined directions that tell the solver exactly how to move objects apart to resolve a collision. This makes the simulation much more stable and differentiable.

### 2. Hydroelastic Contact Modeling

In older simulators, when two objects touched, the engine calculated the force at a single "contact point." This often caused robots to jitter or "explode" because the force concentration was too high.

**Hydroelastic Contact** treats nominally rigid objects as if they have a thin, compliant (soft) layer:
- Instead of a single point, it calculates **distributed pressure** over the entire contact area.
- It uses precalculated pressure fields to simulate the "squish" of surfaces without the massive overhead of full Finite Element Analysis (FEA).

**Why it matters:** This provides realistic friction and "torsional" behavior (like twisting a lid on a jar). For tasks like picking up a small screw or handling fragile objects, hydroelastic modeling is the difference between success and a simulation glitch.

## Official Resources

- **Official Website:** [nvidia.com/newton](https://developer.nvidia.com/newton)
- **GitHub Repository:** [github.com/newton-physics/newton](https://github.com/newton-physics/newton)
- **Documentation:** [newton-physics.github.io](https://newton-physics.github.io)

## Performance Metrics Summary (GTC 2026)

| Task | Platform | Speedup vs MJX | Configuration |
| --- | --- | --- | --- |
| **Humanoid Locomotion** | RTX 6000 Ada (Blackwell) | **252x** | 16,384 envs |
| **Dexterous Manipulation** | RTX 6000 Ada (Blackwell) | **475x** | 8,192 envs |
| **Cloth Simulation** | RTX 6000 Ada (Blackwell) | **120x** | 10k vertices |
| **Cable / Wire Handling** | RTX 6000 Ada (Blackwell) | **185x** | SDF + VBD Solver |

*Note: Benchmarks performed with end-to-end RL loops where tensors remain on-GPU.*

## Quick Start

### 1. Installation

Newton can be installed via `pip`. It is recommended to install with examples for first-time users.

```bash
# Core installation
pip install newton-physics

# Installation with examples and viewer
pip install "newton-physics[examples,viewer]"
```

### 2. Run Built-in Examples

Test your installation by running the basic pendulum or URDF viewer:

```bash
# Run basic pendulum simulation
python -m newton.examples.basic_pendulum

# View a URDF model
python -m newton.examples.basic_urdf --path /path/to/your/robot.urdf
```

### 3. Basic Python Script

A minimal example to initialize a scene and step the simulation:

```python
import newton
import numpy as np

# Initialize engine with CUDA backend
engine = newton.Engine(backend="cuda")

# Create a scene from a USD or URDF file
scene = engine.create_scene("my_robot_scene")
robot = scene.add_robot("unitree_g1.urdf", pos=[0, 0, 0.7])

# Simulation loop
for _ in range(1000):
    # Apply random actions (tensors stay on GPU)
    actions = np.random.uniform(-1, 1, robot.num_dof)
    scene.step(actions)
    
    # Get state without CPU-GPU transfer overhead
    state = scene.get_state()
```