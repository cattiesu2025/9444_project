# Integrated notebook submission

`project.ipynb` is the single integrated notebook. Its six main sections follow the project requirements:

1. Introduction, Motivation and Problem Statement
2. Data Sources
3. Exploratory Analysis of Data
4. Models and Methods
5. Results
6. Discussion, followed by References

Writing quality applies throughout; it is not a separate section. The narrative explains the research questions, methods, measurements and interpretation. Engineering records, job identifiers, hashes and troubleshooting histories belong in supporting files, not the scientific narrative.

## Running the notebook

Use a Python environment with the packages in `requirements-notebook.txt`, plus NumPy, Matplotlib and PyTorch (CPU is sufficient). Restart the kernel and run all cells in order. Results are saved measurements embedded in the notebook; code excerpts and synthetic checks illustrate implementation correctness. Default execution does not train models, download files or require SYSU-MM01 images, checkpoints, a GPU or the repository on the Python path. Figures from external papers and image examples are embedded as notebook attachments.

`project.html` is the corresponding reading preview, not another submission notebook. `evidence/notebook_validation.json` records the latest execution method, notebook hash and output checks. The current environment blocks TCP and IPC kernel sockets. All 48 code cells passed sequential replay in a fresh IPython process and empty working directory, with tables and figures captured; a conventional Jupyter Restart & Run All remains recommended. Validation does not rerun or independently reproduce GPU experiments.

## C contribution integrated on 21 November 2026

- Section 4.3: motivation, PBSVT attribution, exact grayscale schedule, exposure-matched F control, a core excerpt from `src/schedules.py`, synthetic boundary/exposure/reconstruction checks and probability curves.
- Section 5.4: saved three-seed M0/M1 comparisons, F's single-seed control, S-plus-L comparison, interpretation, limitations and the L3_B025 configuration contribution.
- Section 6.2: the answer to RQ1, consistent with the measured negative result.

The notebook narrative, code comments and generated labels are in English. S probability curves are computed from the configuration and are not measured loss or adaptation curves. No missing M1/F training trajectories have been fabricated. F and the initial local-configuration exploration remain explicitly single-seed results. The unavailable Q retrieval result is disclosed as a limitation rather than left as a TODO or invented.

## Supporting code

Build the separate code archive from the submitted revision, as described in Appendix A:

```bash
git archive --format=zip -o submission/project_code.zip HEAD src configs scripts tests requirements-lock.txt requirements-katana-lock.txt requirements-notebook.txt README.md ATTRIBUTION.md
```

Submit that archive only through the supporting-file route allowed by the course announcement. Do not include datasets, pretrained weights or checkpoints. Full retraining requires the repository configuration, dataset and training dependencies; it is separate from replaying the notebook's saved results.

## Evidence and final review

- `docs/evaluation.md`: development and final-test results and evaluation protocol.
- `docs/final_stage.md`: final configuration decisions and 395-identity retraining.
- `docs/baseline.md`: training and restoration details.
- `artifacts/results/`: external run logs, features and provenance records.
- `submission/evidence/`: versioned notebook execution records and compact measurements.

Before the group submits, review the member information, citations, final conclusions and attachment list. The summary report has a separate maximum of four body pages, excluding references. A successful notebook execution does not certify the report, presentation or administrative submission requirements.
