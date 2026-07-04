## Schema-Guided Response Generation using Multi-Frame Dialogue State for Motivational Interviewing Systems

This repository provides the dataset and implementation accompanying our paper "Schema-Guided Response Generation using Multi-Frame Dialogue State for Motivational Interviewing Systems", published in Findings of the Association for Computational Linguistics: ACL 2026.
- https://aclanthology.org/2026.findings-acl.2063/
- pdf: https://aclanthology.org/2026.findings-acl.2063.pdf


### Dataset
- We plan to release a dataset consisting of transcriptions of dialogues between professional counselors and general participants on the topic of eating habits. This dataset will be available for research purposes only, and access will be granted upon request through an application form.


### Code
- `src/` contains the implementation of our pipeline, including dialogue frame extraction, pseudo dialogue strategy generation, dialogue policy pool server, and the dialogue system itself. See [README_dialogue_system.md](README_dialogue_system.md) for setup and usage instructions.

## Citation

If you use this dataset or code, please cite the following paper:

```
@inproceedings{zeng-nakano-2026-schema,
    title = "Schema-Guided Response Generation using Multi-Frame Dialogue State for Motivational Interviewing Systems",
    author = "Zeng, Jie  and
      Nakano, Yukiko",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.2063/",
    doi = "10.18653/v1/2026.findings-acl.2063",
    pages = "41493--41524",
    ISBN = "979-8-89176-395-1",
}
```