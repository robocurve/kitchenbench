# Inspect Robots setup scenes

One `inspect-robots-setup` v1 scene per KitchenBench task, for the Inspect
Robots phone apps (iOS/Android): the app overlays a grid on the physical
bench, the operator taps the two arm-base positions, and the scene's objects
appear as labeled markers at their sampled poses.

The app's format stores anchors as an ordered `references` list of labeled
points; for these scenes the two anchors are the rig's arm bases, so they
carry the labels "LEFT arm" and "RIGHT arm" in the JSON itself. The apps
hardcode no arm wording; the placement prompts ("Tap the LEFT arm base")
come from these labels. Scenes in the older single `reference` object shape
still load, prompting by numbered position instead.

Each file is the task's **first instance** projected through the
`yam-bimanual` rig: epoch 0 is the default realization and epochs 1–4 are the
`variants` the app's **Resample** button draws from — so all five realizations
per instance come from the benchmark's own setup distributions. Only
bench-frame objects are included; parented objects (a lid riding a jar) are
resolved by the task itself, not by table markup.

Like scene layouts, these files are regenerable projections — do not
hand-edit them. Regenerate after changing specs:

```sh
for id in place_cutlery/spoon-on-plate stack/cups place_in_rack/plate \
          pour_pasta/measuring-cup-to-bowl open_container/jar \
          fold_cloth/dish-towel seal_container/food-container handoff/utensil \
          sort_cutlery/balanced-pile scoop_pasta/spoon-penne; do
  for epoch in 0 1 2 3 4; do
    kitchenbench-layout "$id" --epoch "$epoch"
  done | jq -s --slurpfile rig rigs/yam-bimanual.json '
    def objs: [.objects[] | select(.frame=="bench")
               | {label:.asset, xy_cm:(.xy_cm|map(.*100|round/100))}
               + (if .yaw_deg != 0 then {yaw_deg:(.yaw_deg*100|round/100)} else {} end)];
    def friendly: split("/")
      | ((.[0]|gsub("_";" ")|(.[0:1]|ascii_upcase)+.[1:]) + ": " + (.[1]|gsub("-";" ")))
      + " · KitchenBench";
    {format:"inspect-robots-setup", version:1, name:(.[0].instance_id|friendly),
     instruction:.[0].instruction,
     references:[{label:"LEFT arm", xy_cm:$rig[0].arms.left.base_xy_cm},
                 {label:"RIGHT arm", xy_cm:$rig[0].arms.right.base_xy_cm}],
     objects:(.[0]|objs),
     variants:[.[1:][] | {instruction:.instruction, objects:objs}]}' \
    > "scenes/inspect-robots/${id%%/*}.json"
done
```

The scene format itself is documented in the Inspect Robots app repo
(`SETUP_FORMAT.md`).
