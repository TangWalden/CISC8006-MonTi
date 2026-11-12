# Supplementary author-checkpoint staged evaluation

Status: source prepared; strict checkpoint compatibility and TRAIN preflight
have not run. This source does not replace or modify the active formal study.

One official `GCN-GAGA-gossipcop_0.05-best_model.pickle` is frozen throughout.
Its SHA256 is `9f5dc7dd594a8e8927449665ef3a07b777199c2436d6fd1cc1f7adf0fdb6c23d`;
the corresponding 737-byte configuration has SHA256
`e27fd6959f35ab1786f35e4e70c89bec5156da1a6577f4481287310ebf552c2a`.
The [official MonTi repository](https://github.com/bdi-lab/MonTi) links this
model through its
[Google Drive release folder](https://drive.google.com/drive/folders/1E2FE8zDIS4XZA6Fxe5l3hTrXUnfhJ2gr),
in `best_models.zip`. The retained original setup instructions are available in
the [historical upstream README](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/docs/UPSTREAM_README.md).
Acquisition verified HTTP ranges, ZIP headers, size and CRC; the author did not
supply a signed release or independent SHA256. Object compatibility is a
separate registered preflight check.

The existing metadata for all 937 TRAIN groups and 909 TEST groups is inherited from the frozen
formal registration without reopening partition arrays or selecting groups.
The three arms remain native author MonTi, projected-quota random completion,
and projected-quota trained completion. The scorer, four terminal epochs,
quota policy, total structural budgets, original GCN feedback and final delayed
GAGA evaluation are unchanged. The upstream parameters never update.

Author configuration seed 3 remains truthful in all three repetitions.
Downstream initialization/order and per-case upstream latent-noise repetitions
use seeds 0, 1 and 2. Capture epoch 0 is a synthetic RNG key, not an author best epoch.
Inference starts fresh temperature/epsilon schedulers at 10 with zero steps.
The native eval path ignores their values for Gumbel randomness, but still
draws auxiliary latent noise when more than one node is injected. This is an
explicitly seeded paired study, not bitwise replay of the authors' CLI.

Thus uncertainty is conditional on ONE released generator plus three
downstream/capture-noise repetitions; it does not measure variability across
three independently trained author generators. Paper-table percentages and
our earlier own-trained baselines are context, never substituted paired arms.
The primary rule remains positive mean, all three gains positive and nominal
paired t 95% lower endpoint above zero. Random completion is secondary, not a
replacement winner. Repeated prior TEST inspection remains disclosed.

After committed source/auditor freeze, `experiments/author_staged/prepare.py`
creates the explicit `monti_author_staged_v1` registration. The reused guardian
dispatches with `author_staged_exclusive.lock`; unknown or cross-profile locks
are rejected. Preflight/run use the existing final-staged algorithms through
the declared author loader. Four CPU threads, a 12-hour main limit, a 16-hour phase limit, no automatic retries
and all previous logical query caps remain fixed. All final plans/weights must
seal before official GAGA loads. Human review and final submission stay separate.
