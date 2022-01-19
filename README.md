# PICUNeuroPrediction

Admission predictive modeling for PICU patients with suspected acute neurological injury

## Data Source

Trichotomous Outcomes in Pediatric Critical Care (TOPICC, https://www.cpccrn.org/study-datasets/topicc/). Due to data use restrictions with our agreement with the NICHD, we are unable to share the primary data source. However, the data
source is found at the NICHD Data and Specimen Hub (DASH, https://dash.nichd.nih.gov/study/226509) and qualified investigators can obtain it
independently; no modifications were made to the data prior to analysis.

## Model

Code to generate Random Forest, Gradient Boosting, Neural Network (multi-level perceptron), SVM, logistic regression, and ensemble
models to predict mortality or morbidity/mortality from TOPICC data. Recommended installation: virtual Python 3 environment with packages
of pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels, imblearn. Explainability requires 'shap'. Causal analysis uses causallearn and networkx.

## FHIR server

Code provided to provide FHIR access and process EHR data to TOPICC-equivalent
row to feed into pre-trained Random Forest models and show explanations.
Requires fhirpy, shap, joblib, matplotlib, pandas, numpy libraries. Working
installation found at https://fhirdemo.nkmj.org. Installation there includes
nginx, uwsgi, flask. FHIR server should be agnostic, but live installation uses
HAPI FHIR, (https://hapifhir.io/).
