# spaCy-ltp

---

This repo is inspired by [spaCy-stanza](https://github.com/explosion/spacy-stanza)

In this repo, we wraps [LTP 4](https://github.com/HIT-SCIR/ltp)(Language Technology Platform) library, so you can
use `LTP` models (including 'cws', 'pos', 'dep' and 'ner') in a [spaCy](https://github.com/explosion/spaCy) pipline

Using this wrapper, you'll be able to use the following tasks, computed by your pretrained `LTP` model:

- Chinese word segmentation (cws): `Doc` and its `tokens`
- Part-of-speech tagging (pos): `token.tag_`
- Dependency parsing (dep): `token.dep_`, `token.head`
- Named entity recognition (ner): `doc.ents`, `token.ent_type`

## Requirements

```text
spacy v3.x
ltp v4.2.x
```

## Usage & Example

```python
from ltp import LTP
import spacy_ltp

# [Optional] Download the LTP model if necessary
LTP('LTP/small')

# Initialize the pipeline
nlp = spacy_ltp.load_pipeline('LTP/small')

doc = nlp("华东师范大学是教育部和上海市人民政府重点共建的综合性研究型全国重点大学。")
for token in doc:
    print(token.text, token.tag_, token.dep_, token.head, token.ent_type_) 
```

| Text | TAG | DEP | HEAD | ENT |
|-----|-----|-----|------|-----|
|华东   |ns |ATT |师范大学 |Ni |
|师范大学 |n  |SBV |是    |Ni |
|是    |v  |HED |是    |   |
|教育部  |ni |SBV |共建   |Ni |
|和    |c  |LAD |人民政府 |   |
|上海市  |ns |ATT |人民政府 |Ni |
|人民政府 |i  |COO |教育部  |Ni |
|重点   |d  |ADV |共建   |   |
|共建   |v  |ATT |大学   |   |
|的    |u  |RAD |共建   |   |
|综合性  |n  |ATT |大学   |Ni |
|研究型  |b  |ATT |大学   |Ni |
|全国   |n  |ATT |大学   |Ni |
|重点   |n  |ATT |大学   |Ni |
|大学   |n  |VOB |是    |Ni |
|。    |wp |WP  |是    |   |

```python
for ent in doc.ents:
    print(ent.text, ent.label_)
```

| ENT          | TYPE |
|--------------|------|
| 华东师范大学       | Ni   |
| 教育部          | Ni   |
| 上海人民政府       | Ni   |
| 综合性研究型全国重点大学 | Ni   |




