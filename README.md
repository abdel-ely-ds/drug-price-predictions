# Drug Price Prediction

## Project Description

The objective is to predict the price for each drug in the test data set (`drugs_test.csv`). Please refer to the `sample_submission.csv` file for the correct format for submissions.

## Guidelines

Build a machine learning pipeline to train a model and generate predictions for the test set. We expect an application layout, not notebooks (but feel free to also share your notebooks if you want). Structure your code so that it can be packaged and deployed to production. Think that your code will be the first iteration of a pipeline the company will use in production.

You are free to define appropriate performance metrics that fit the problem and chosen algorithm.

Please modify `README.md` to add:

1. Instructions on how to run your code.
2. A paragraph or two about what algorithm was chosen for which problem, why (including pros/cons) and what you are particularly proud of in your implementation, and why.
3. Overall performance of your algorithm(s).

## Evaluation criteria

- **Code quality**: code is written once but read many times. Please make sure that your code is well-documented, and is free of programmatic and stylistic errors.
- **Reproducibility and replicability**: We should be able to reproduce your work and achieve the same results.

Evaluation of your submission will be based on the following criteria:

1. Did you follow the instructions for submission?
2. Can we package and deploy your code to production?
3. Did you apply an appropriate machine learning algorithm for the problem and why you have chosen it?
4. What features in the data set were used and why?
5. What design decisions did you make when designing your models? Why (i.e. were they explained)?
6. Did you separate any concerns in your application? Why or why not?

There are many ways and algorithms to solve these questions; we ask that you approach them in a way that showcases one of your strengths. We're happy to tweak the requirements slightly if it helps you show off one of your strengths.

## Files & Field Descriptions

You'll find five CSV files:
- `drugs_train.csv`: training data set,
- `drugs_test.csv`: test data set,
- `active_ingredients.csv`: active ingredients in the drugs.
- `drug_label_feature_eng.csv`: feature engineering on the text description,
- `sample_submission.csv`: the expected output for the predictions.

### Drugs

Filenames: `drugs_train.csv` and `drugs_test.csv`

| Field | Description |
| --- | --- |
| `drug_id` | Unique identifier for the drug. |
| `description` | Drug label. |
| `administrative_status` | Administrative status of the drug. |
| `marketing_status` | Marketing status of the drug. |
| `approved_for_hospital_use` | Whether the drug is approved for hospital use (`oui`, `non` or `inconnu`). |
| `reimbursement_rate` | Reimbursement rate of the drug. |
| `dosage_form` | See [dosage form](https://en.wikipedia.org/wiki/Dosage_form).|
| `route_of_administration` | Path by which the drug is taken into the body. Comma-separated when a drug has several routes of administration. See [route of administration](https://en.wikipedia.org/wiki/Route_of_administration). |
| `marketing_authorization_status` | Marketing authorization status. |
| `marketing_declaration_date` | Marketing declaration date. |
| `marketing_authorization_date` | Marketing authorization date. |
| `marketing_authorization_process` | Marketing authorization process. |
| `pharmaceutical_companies` | Companies owning a license to sell the drug. Comma-separated when several companies sell the same drug. |
| `price` | Price of the drug (i.e. the output variable to predict). |

**Note:** the `price` column only exists for the train data set.

### Active Ingredients

Filename: `active_ingredients.csv`

| Field | Description |
| --- | --- |
| `drug_id` | Unique identifier for the drug. |
| `active_ingredient` | [Active ingredient](https://en.wikipedia.org/wiki/Active_ingredient) in the drug. |

**Note:** some drugs are composed of several active ingredients.

### Text Description Feature Engineering

Filename: `drug_label_feature_eng.csv`

This file is here to help you and provide some feature engineering on the drug labels.

| Field | Description |
| --- | --- |
| `description` | Drug label. |
| `label_XXXX` | Dummy coding using the words in the drug label (e.g. `label_ampoule` = `1` if the drug label contains the word `ampoule` - vial in French). |
| `count_XXXX` | Extract the quantity from the description (e.g. `count_ampoule` = `32` if the drug label  the sequence `32 ampoules`). |

**Note:** This data has duplicate records and some descriptions in `drugs_train.csv` or `drugs_test.csv` might not be present in this file.

Good luck.
