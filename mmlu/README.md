# MMLU benchmark data

Official MMLU (Massive Multitask Language Understanding) data, downloaded from the
canonical Berkeley tarball linked by the paper's repository:

- Paper: *Measuring Massive Multitask Language Understanding* (Hendrycks et al., ICLR 2021)
- Repo: <https://github.com/hendrycks/test>
- Data: <https://people.eecs.berkeley.edu/~hendrycks/data.tar>
- Downloaded: 2026-10-02

## Source files

- `data.tar` — the original downloaded archive (159 MB).
- `data/` — extracted contents.

## Layout

```
data/
  test/           57 csv files, 14,042 questions  <- the evaluation set
  val/            57 csv files,  1,531 questions  <- model selection (not for training)
  dev/            57 csv files,    285 questions  <- 5-shot examples for prompting
  auxiliary_train/  8 csv files, ~99,947 rows     <- optional fine-tuning data
  README.txt
  possibly_contaminated_urls.txt
```

Each CSV row is:

```
question, A, B, C, D, answer
```

- `question` may contain commas and is quoted when needed.
- `A`-`D` are the four answer choices.
- `answer` is one of `A`, `B`, `C`, `D`.

Parse with a real CSV reader (fields can contain commas/quotes), e.g.:

```python
import csv
with open("data/test/abstract_algebra_test.csv", newline="", encoding="utf-8") as f:
    for question, a, b, c, d, answer in csv.reader(f):
        ...
```

Use `data/dev/<subject>_dev.csv` for the standard 5-shot prompt examples and
`data/test/<subject>_test.csv` for scoring. There are 57 subjects; subject names
are the filename stem before `_test` / `_val` / `_dev`.
