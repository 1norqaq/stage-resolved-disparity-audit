# Ethics and Privacy Statement

The audit concerns a high-stakes employment context and uses sensitive personal data. The repository is structured to minimize risk while allowing independent review of the statistical workflow.

## Risk-sensitive features

The governed data include or derive from:

- applicant names;
- CV content;
- geolocation and demographic fields;
- inferred origin categories;
- candidate-job fit representations;
- status changes through the hiring funnel;
- final hiring outcomes.

## Mitigations used in the released materials

- No real row-level applicant data are included.
- No names, CV text, geolocation coordinates, employer identifiers, or linkage keys are included.
- The released code reads an already-linked analysis table; it does not expose the private bridge used for record linkage.
- Synthetic data are generated from random distributions and do not encode real applicant records.
- Output tables should be reviewed for small-cell disclosure before public release.

## Recommended handling of governed data

- Store governed files outside the public repository.
- Use encrypted storage or a controlled research environment.
- Keep access logs for all governed files.
- Do not export row-level intermediate files unless necessary.
- Suppress or coarsen any aggregate cell below the disclosure threshold set by the data controller.
- Do not attempt to re-identify applicants.
- Do not use inferred protected attributes for operational hiring decisions.

## Interpretation safeguards

The analysis estimates disparate impact and stage-level attribution within an observational audit. It does not establish individual intent, and it should not be used to make claims about any named applicant, recruiter, or team. Results should be reported at aggregate group and stage levels only.
