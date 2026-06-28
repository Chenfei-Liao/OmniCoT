# OmniCoT lmms-eval Task

This directory adds local OmniCoT evaluation tasks to the vendored `lmms-eval`
copy.

Default bundled smoke-test data:

```text
lmms_eval/tasks/omnicot/sample_data/OmniCoT_sample.json
lmms_eval/tasks/omnicot/sample_data/image/
```

The bundled sample contains six real OmniCoT cases, one for each released
question type, plus the corresponding ERP panoramic image. It is intended for
installation and pipeline smoke testing.

For full benchmark evaluation, replace `dataset_kwargs.data_files.test` in
`_default_template.yaml` with the full OmniCoT JSON path, or copy the full JSON
into this directory and set `OMNICOT_DATA_DIR` / `OMNICOT_IMAGE_DIR` to matching
relative data folders in your environment.

Example:

```bash
python -m lmms_eval \
  --tasks omnicot_no_desc \
  --limit 8 \
  --output_path results/omnicot_smoke \
  ...
```

The answer-evaluation pass writes `submissions/omnicot_submission.json` under
the selected `--output_path`. This file is the input for CoT-quality evaluation:

```bash
python tools/omnicot_cot_quality_eval.py \
  --input-file results/omnicot_smoke/submissions/omnicot_submission.json \
  --output-dir results/omnicot_smoke/cot_quality \
  --mode all
```

CoT-quality modes:

- `simple`: precision, recall, and F1 against the reference CoT.
- `spatial`: viewpoint consistency, spatial evidence sufficiency, and reasoning feasibility.
- `all`: both metric groups.
