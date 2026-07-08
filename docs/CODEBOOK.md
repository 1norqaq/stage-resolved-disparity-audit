# Codebook

The analysis table is one row per application (`candidate_id` × `job_id`). Column names below are the canonical names expected by the scripts.

## Identifiers

| Column | Type | Description |
|---|---:|---|
| `application_id` | string | Stable application identifier. |
| `candidate_id` | string | Stable candidate identifier. Used for clustered standard errors and bootstrap resampling. |
| `job_id` | string | Job posting identifier. |

## Group attributes

| Column | Type | Description |
|---|---:|---|
| `ethnicity` | categorical | Inferred origin group. Reference category: `French`. |
| `ethnicity_posterior` | float | Top-class posterior confidence for the inferred origin label. |
| `gender` | categorical | Declared or recorded gender where available. Missing values should be coded as `Unknown`. |
| `age_band` | categorical | Age bucket, if available. |
| `region` | categorical | Candidate or application region. |

## Qualification and content features

| Column | Type | Description |
|---|---:|---|
| `fit_q` | float | Candidate-job embedding cosine similarity or normalized fit score. |
| `experience_duration` | float | Total work-experience duration. Unit should be documented in `docs/DATA_AVAILABILITY.md`. |
| `education_duration` | float | Total education duration. Unit should be documented in `docs/DATA_AVAILABILITY.md`. |
| `text_length` | integer | Length of parsed CV text after the preprocessing used in the paper. |

## Job, time, and sorting controls

| Column | Type | Description |
|---|---:|---|
| `job_family` | categorical | High-level job family. |
| `job_category` | categorical | More granular job category. |
| `application_region` | categorical | Region of the job, candidate, or application, depending on the audit export. |
| `application_month` | categorical | Month of application or status-log entry in `YYYY-MM` format. |

## Outcomes

| Column | Type | Description |
|---|---:|---|
| `advanced` | integer/bool | 1 if the application advanced past the CV screen; 0 otherwise. |
| `hired` | integer/bool | 1 if the application reached hire; 0 otherwise. |
| `prequalified` | integer/bool | 1 if the application reached or passed the pre-qualification milestone. |
| `interview_shortlist` | integer/bool | 1 if the application reached interview or shortlist. |
| `offered` | integer/bool | 1 if an offer was proposed or the application reached the offer milestone. |

## Optional embedding columns

Representation-probe scripts expect profile-embedding columns named:

```text
emb_0, emb_1, ..., emb_1023
```

If these columns are absent, scripts that require embeddings will skip the probe with a clear message.

## Missing values

- `candidate_id`, `ethnicity`, `fit_q`, and `advanced` are required for the screen-stage analysis.
- `hired` is required for the hire-gate analysis.
- Missing categorical controls should be explicitly coded as `Unknown` before modeling.
- Missing numeric covariates should either be imputed upstream or excluded using a documented complete-case rule.

## Reference categories

- Origin reference: `French`.
- Gender reference: the first non-missing value after sorting unless explicitly set in the script. For the paper analysis, the gender coefficient is interpreted after conditioning on origin and fit, not as a headline claim.
