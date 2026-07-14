# KitchenBench

KitchenBench is 10 kitchen-manipulation tasks expressed as
[Inspect Robots](https://github.com/robocurve/inspect-robots) `Task`s. They are
embodiment-agnostic, so you run them against any compatible policy/embodiment
pair. The set emphasizes bimanual coordination: pouring, lid removal, folding,
part-mating, a pure two-arm handover, and tool-mediated scooping, alongside
classic pick-place / stacking / slotted insertion and a multi-instance sort.

This page is the reference for what each task asks and everything it varies
over. For installation, the quick start, and running on real hardware or in
simulation, see the
[README on GitHub](https://github.com/robocurve/kitchenbench#readme).

## The tasks

| Task (`--task`) | Goal | Varies over | Bimanual | Category |
|---|---|---|:--:|---|
| `kitchenbench/place_cutlery` | place the {cutlery} on the {dishware} | {cutlery}: spoon, fork, knife · {dishware}: plate, bowl, napkin · placement, approach angle, clutter | | pick-place |
| `kitchenbench/stack` | stack the cups / bowls / plates | stacked item: cups, bowls, plates · count: 2 to 5 · sizes, spacing, placement jitter | | stacking |
| `kitchenbench/place_in_rack` | place the {dishware} into the dish rack | {dishware}: plate, bowl, cup · slot, rack pose, approach tilt, slot width, friction | | insertion |
| `kitchenbench/pour_pasta` | pour the dry pasta into the {vessel} | {vessel}: bowl, cup, pot · fill amount, pour height and angle, placement | ✅ | granular |
| `kitchenbench/open_container` | open the {container} | {container}: jar, bottle, food container · lid torque, cap turns, tilt, size | ✅ | articulated |
| `kitchenbench/fold_cloth` | fold the {cloth} | {cloth}: dish towel, napkin, cloth · size, fold count, crumple, rotation | ✅ | deformable |
| `kitchenbench/seal_container` | seal the {container} with its lid | {container}: food container, pot, jar · lid offset and yaw, press force, thread turns, fit clearance | ✅ | mating |
| `kitchenbench/handoff` | hand off the {item} from one arm to the other | {item}: utensil, cup, produce item · handoff height and position, fill level, fragility, tool length | ✅ | coordination |
| `kitchenbench/sort_cutlery` | sort the cutlery into the correct tray compartments | spoons, forks, knives · piece count: 3 to 10 · pile spread and overlap, tray pose | | classification |
| `kitchenbench/scoop_pasta` | scoop about {fill_target_g} g of the {pasta} with the {tool} and transfer it to the container | {pasta}: penne, rigatoni · {tool}: spoon, measuring cup · {fill_target_g}: 20 to 160 · pile depth, container distance | ✅ | granular+tool |

The "Varies over" column is the union over the task's 5 instances. Each
`{placeholder}` lists every value the goal sentence can name (an individual
instance samples from a subset of them and fixes the rest in its goal text);
the trailing items summarize the remaining setup variables (placement jitter,
sizes, forces, and the like) the instances also sample, without listing every
one. The full per-instance distributions live in
[`specs.py`](https://github.com/robocurve/kitchenbench/blob/main/src/kitchenbench/specs.py),
and `tests/test_docs_table.py` keeps this table in sync with them.

## Where things live

- [GitHub repository](https://github.com/robocurve/kitchenbench): source,
  quick start, hardware and simulation guides
- [PyPI package](https://pypi.org/project/kitchenbench/): `pip install kitchenbench`
- [WorldEvals](https://github.com/robocurve/worldevals): the benchmark
  collection KitchenBench belongs to
- [Physical-automation methodology](https://github.com/jeqcho/physical-automation-methodology-docs):
  the instance/realization model the tasks follow
