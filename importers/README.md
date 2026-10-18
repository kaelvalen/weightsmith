# importers (F5 · pretrained import)

Not part of the MVP (F0–F3). Planned scope, fixed on purpose:

- Read an HF/PyTorch model as a module tree and translate the supported families
  (Llama-style decoder, ViT) into graph IR.
- Anything that does not translate (dynamic control flow, weight tying, custom
  `forward`) becomes an **opaque block** node: its inside is not shown, but it is
  connectable and LoRA can target its linear sub-modules.
- Adding a family is separate work; the supported list does not grow by accident.

The IR, codegen (LoRA targets by sub-module path) and the Subgraph node are already
shaped for this; an importer only has to emit IR.
