# Examples

## `notebooks/semsimula_diag_tour.ipynb`

A guided tour of every utility in the package: how to configure it, what it
measures, and when to reach for it. Each code cell is preceded by a markdown
cell explaining the *why*, not just the *how*.

You supply four paths at the top (checkpoint directory, archive root,
checkpoint prefix, and a step number); everything else is derived.

Sections 1–8 need only a bundle file and run on CPU in seconds. Section 9
(the stiffness probes) needs a built model with an anisotropic Gaussian
`V_theta` — supply your own in the marked cell; the rest of the notebook
still runs if you skip it.

Covered: `BundleStore`, bundle anatomy, `GradClipConfig` (including the
override-ordering trap), `per_group_grad_norms` vs `clip_grads_per_group`,
`ClipThenSum`, `isolated_grads`, `ProbeContext`, the token probes, the five
stiffness probes, and `ProbeResult`.
