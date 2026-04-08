# Hardware & Robotics Ecosystem

This document catalogs open-source and low-cost robotic hardware designs, assembly guides, and resources that complement the simulation benchmarks.

---

## [XLeRobot](https://xlerobot.readthedocs.io/en/latest/index.html)
A practical, low-cost (~$660) dual-arm mobile robot design for general household manipulation, built upon the [LeRobot](https://github.com/huggingface/lerobot) foundation and other open source communities like SO-100.

### Key Features
- **Affordable Cost**: Complete basic build costs around $660, making it cheaper than an iPhone.
- **Easy Assembly**: ~90% 3D printed components. The two SO-101 arms can be assembled and configured in under 4 hours.
- **Plug-and-Play Software**: Simple `pip install` approach, directly compatible with LeRobot's extensive Vision-Language-Action (VLA) model library and dataset ecosystem.
- **Mobile Base Enhancement**: Addresses the durability and load capacity issues of DIY mobile robots by providing a more stable, practical chassis and power supply solution for daily use.
- **Safe & Practical**: Capable of completing many household chores while retaining inherent safety due to its lightweight frame and low-torque motors.

### Resources
- [Documentation](https://xlerobot.readthedocs.io/en/latest/index.html)
- [GitHub Repository](https://github.com/Vector-Wangel/XLeRobot)

---

## [ASTORINO](https://astorino.com.pl/en/)
A 6-axis educational industrial robot designed for vocational schools and technical universities, manufactured by ASTOR (Kawasaki Robotics CEE Hub).

### Key Features
- **Industry-Standard Learning**: Programmed exactly like real Kawasaki Robotics industrial robots, providing direct and highly transferable career skills for students.
- **3D Printed Foundation**: 99% of the structural components are 3D printed (using popular PET-G filament). STL files are supplied with the robot so parts can be self-repaired or reprinted.
- **Safe & Didactic Design**: Features built-in brakes on the 2nd and 3rd axes (fall protection), collision detection, and a safe payload envelope suitable for student labs.
- **Expandable Modularity**: Integrated with an internal microcontroller, accepting external 24V I/O modules, vision systems, cube feeders, various grippers, and even a 7th axis rail.
- **Ecosystem Formats**: Ranges from bare robot configurations to "Station PRO" setups featuring dedicated worktables and pneumatic grippers.

### Resources
- [Product Website](https://astorino.com.pl/en/)

---

## [Universal Manipulation Interface (UMI)](https://umi-gripper.github.io/)
We present Universal Manipulation Interface (UMI) -- a data collection and policy learning framework that allows direct skill transfer from in-the-wild human demonstrations to deployable robot policies. UMI employs hand-held grippers coupled with careful interface design to enable portable, low-cost, and information-rich data collection for challenging bimanual and dynamic manipulation demonstrations.

To facilitate deployable policy learning, UMI incorporates a carefully designed policy interface with inference-time latency matching and a relative-trajectory action representation. The resulting learned policies are hardware-agnostic and deployable across multiple robot platforms.

Equipped with these features, the UMI framework unlocks new robot manipulation capabilities, allowing zero-shot generalizable dynamic, bimanual, precise, and long-horizon behaviors, by only changing the training data for each task. We demonstrate UMI’s versatility and efficacy with comprehensive real-world experiments, where policies learned via UMI zero-shot generalize to novel environments and objects when trained on diverse human demonstrations.

**Video Demonstration:**
https://umi-gripper.github.io/videos/teaser_data_collection_dish.mp4
