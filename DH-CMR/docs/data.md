# Data preparation

Provide the released dataset features under the following layout, or override `data.path` and `data.text_model`:

```text
datasets/
  mosi/unaligned_50.pkl
  mosei/unaligned_50.pkl
  ch_sims/unaligned_50.pkl
pretrained/
  bert-base-uncased/
  bert-base-chinese/
```

Each pickle is a dictionary with `train`, `valid`, and `test` dictionaries. Extra top-level metadata is ignored. Each split contains:

| Field | Format |
| --- | --- |
| `id` | Unique identifier per sample within the split |
| `raw_text` | Raw utterance strings |
| `audio` | NumPy array `[N, time, features]`, or a list of 2D NumPy sequences |
| `vision` | NumPy array `[N, time, features]`, or a list of 2D NumPy sequences |
| `regression_labels` | One finite multimodal sentiment score per sample |
| `audio_lengths` | Optional valid lengths |
| `vision_lengths` | Optional valid lengths |

Change `data.label_key` if the multimodal label field uses another name. No modality-specific labels are used as model inputs. Already-tokenized text features alone are insufficient: raw text is tokenized using the selected local BERT tokenizer.

Audio and vision dimensions are inferred from training data and checked across all splits. NaN and infinite feature values are replaced by zero. Normalization statistics use valid training frames only and are saved in checkpoints. Evaluation reloads those statistics.

The default length policy retains the greater of the declared length and the last nonzero frame index. Set `data.length_policy=declared` when the preprocessing release defines authoritative lengths. Empty sequences are rejected.

Split-qualified identifiers prevent accidental test/reference ID collisions. Training samples that are also selected reference nodes are excluded from their own retrieved local and global node sets.

MOSI and MOSEI use sentiment scores on the [-3, 3] scale. CH-SIMS uses [-1, 1]. Acc-7 rounds clipped scores to the nearest integer. Acc-5 uses the configured boundaries `[-0.7, -0.1, 0.1, 0.7]`, assigning boundary values to the lower interval. Binary neutral-included evaluation uses `score >= 0`; neutral-excluded evaluation removes ground-truth zeros and uses `score > 0`.

The manuscript split sizes are MOSI 1284/229/686, MOSEI 16326/1871/4659, and CH-SIMS 1368/456/457. The loader reports actual sizes without silently filtering examples to force a count. Confirm the feature release and filtering protocol if counts differ.

CSV metadata is not required for training. Pickle inputs must use built-in containers and supported NumPy array serialization. Model checkpoints should come from a trusted training run.
