# Neural-Network-Challenge-2

<div align='center'>
    <img src='https://images.pexels.com/photos/7163380/pexels-photo-7163380.jpeg' height='350', title='An employee feeling pressured and overwhelmed with her job and the people around her (image courtesy of Pexels)', alt='A woman with a pained expression sits at her work desk and holds her head in her hands, while four other employees or superiors surround her and seemingly overwhelm her with input and instructions' />

*Attrition*[^1]

**Module 19 Challenge**
</div>

## Table of Contents

* [Overview](#Overview)
* [Execution](#Execution)

---

## Overview

### *The Assignment*

The Module 19 Challenge gave us the task of creating a neural network that the HR department of a company could use to predict whether employees were likely to leave. Additionally, HR wanted to predict if some employees may be better suited for work in other departments. Our job was to create a branching neural network to make both of these predictions.

### *Written in*

Jupyter Notebook using Python v3.10.13, Pandas, scikit-learn, pathlib, and TensorFlow/Keras

### *Accessing the notebook*

To access the spam detector notebook, simply download the `.ipynb` file `attrition.ipynb` into a local directory, then load the `attrition.ipynb` file into Jupyter Notebook through your terminal.

---

## Execution

### *How to use*

Simply `Run All Cells` to execute the code in every cell of the notebook in sequence. Alternatively, each cell may be `Run` on its own, though it is still recommended to run them in order.

### *Breaking down the code*

Below is a more in-depth explanation of the various cells coded within the `attrition.ipynb` notebook.

| Cell | Notes[^2] |
| ---: | :--- |
| 1[^3] | Importing libraries and dependencies <br> <br> Importing data to the DataFrame (DF) `attrition_df` <br> <br> Displaying a sample of the data |
| 2[^3] | Determining the number of unique values for every column of `attrition_df` |
| 3 | Declaring `y_df` as subset of `attrition_df` with target columns for later modeling  <br> <br> Displaying a sample of the selected data |
| 4 | Declaring `cols_X` as a list of columns to use for the `X` datasets <br> *Note: The instructions specified "at least 10 columns", so I selected all non-target columns* <br> <br> *Unused: Retained column selection to match original prompted output for reference if less features were needed overall* <br> <br> Declaring `X_df` as a subset of `attrition_df` based on `cols_X` <br> <br> Verifying the data types for `X_df` |
| 5 | Splitting the data into training and testing datasets based on `X_df` and `y_df` <br> *Note: Training and testing* `y` *datasets based on* `Attrition` *and* `Department` *features selected earlier* |
| 6 | Declaring `cols_to_encode` as a list of columns to encode from `X_df` <br> <br> Confirming value counts for unique values in `cols_to_encode` |
| 7[^4] | Declaring `cols_for_ohe` as a list of columns for use with `OneHotEncoder` for the `X` datasets <br> <br> Declaring `cols_for_le` as a column for use with `LabelEncoder` for the `X` datasets |
| 8[^4] | Declaring `encoder_X_ohe` as an instance of `OneHotEncoder` for use with `cols_for_ohe` <br> <br> Fitting `encoder_X_ohe` to `cols_for_ohe` in `X_train` and transforming `cols_for_oh` in `X_train` as declared `X_train_ohe` and `X_test` as declared `X_test_ohe` <br> <br> Converting results to DFs `X_train_ohe` and `X_test_ohe` <br> <br> Displaying a sample of `X_train_ohe` to confirm conversion |
| 9[^4] | Declaring `encoder_X_le` as an instance of `LabelEncoder` for use with `cols_for_le` <br> <br> Fitting `encoder_X_le` to `cols_for_le` in `X_train` and transforming `cols_for_le` in `X_train` as declared `X_train_le` and `X_test` as declared `X_test_le` <br> <br> Converting results to DFs `X_train_le` and `X_test_le` <br> <br> Displaying a sample of `X_train_le` to confirm conversion |
| 10[^4] | Declaring `cols_to_scale` as a list of columns for use with `StandardScalar` for the `X` datasets |
| 11 | Declaring `scalar_X` as an instance of `StandardScalar` for use with `cols_to_scale` <br> Fitting `scalar_X` to `cols_to_scale` in `X_train` and transforming as declared `X_train_scaled` <br> <br> Transforming `cols_to_scale` in `X_test` as declared `X_test_scaled` <br> <br> Converting results to DFs `X_train_scaled` and `X_test_scaled` <br> <br> Displaying a sample of `X_train_scaled` to confirm conversion |
| 12[^4] | Concatenating scaled and encoded `X` datasets back into `X_train` and `X_test` <br> <br> Confirming total records in `X` datasets matches total records in the original DF `attrition_df` |
| 13[^4] | Converting `y` datasets to DFs `y_atrn_train_df`, `y_atrn_test_df`, `y_dept_train_df`, and `y_dept_test_df` for use with `OneHotEncoder` <br> *Note: May be unnecessary with refactoring, but encoder would not functions with datasets as arrays* <br> <br> Displaying a sample of `y_atrn_train_df` to confirm conversion to DF |
| 14 | Declaring `encoder_y_dept` as an instance of `OneHotEncoder` for use with `y_dept_train_df` <br> <br> Fitting `encoder_y_dept` to `y_dept_train_df` and transforming as declared `y_dept_ohe_train` <br> <br> Transforming `y_dept_test_df` as declared `y_dept_ohe_test` <br> <br> Displaying `y_dept_ohe_train` to confirm conversion |
| 15 | Declaring `encoder_y_atrn` as an instance of `OneHotEncoder` for use with `y_atrn_train_df` <br> <br> Fitting `encoder_y_atrn` to `y_atrn_train_df` and transforming as declared `y_atrn_ohe_train` <br> <br> Transforming `y_atrn_test_df` as declared `y_atrn_ohe_test` <br> <br> Displaying `y_atrn_ohe_train` to confirm conversion |
| 16 | Declaring `input_shape` as a tuple with the number of columns in `X_train` <br> <br> Creating `input_layer` as the input layer for the model, using `shape=input_shape` <br> *Note: Model built with* `Keras functional API`*, with each layer calling the previous* <br> <br> Creating three (3) Dense layers (`shared_dense_1`, `shared_dense_2`, and `shared_dense_3`) to be shared through the model, all with `activation='relu'` and `units=` set to **decreasing powers of 2** starting from **128** |
| 17 | Declaring `output_shape_dept` as the number of categories in `y_dept_ohe_train` <br> <br> Creating `dept_dense` as a Dense layer, with **16** units and `activation='relu'` <br> <br> Creating `dept_output` as an output layer, named `output_department`, with `output_shape_dept` units and `activation='softmax'` |
| 18 | Declaring `output_shape_atrn` as the number of categories in `y_atrn_ohe_train` <br> <br> Creating `atrn_dense` as a Dense layer, with **16** units and `activation='relu'` <br> <br> Creating `atrn_output` as an output layer, named `output_attrition`, with `output_shape_atrn` units and `activation='sigmoid'` |
| 19 | Creating the model as `model`, named `model` <br> <br> Compiling the model <br> *Optimizer set as* `'adam'` <br> *Loss set as* `categorical_crossentropy` *for* `output_department` *and* `binary_crossentropy` *for* `output_attrition` <br> *Metrics set as* `accuracy` *,* `f1_score` *, and* `precision` *for* `output_department` *and as* `accuracy` *,* `recall` *, and* `precision` *for* `output_attrition` <br> <br> Summarizing the model `model` |
| 20 | Training the model `model` <br> *Note: epochs set to* **100** *, batch size to* **32** *, and validation split to* **0.2** |
| 21 | Declaring `test_results` as an evaluation the model's performance <br> <br> Displaying `test_results` |
| 22 | Printing the accuracy scores for the model's predictions on `Department` and `Attrition` <br> *Note: rounded to four (4) digits for readability* |
| 23[^5] | Printing the remaining declared metrics for the model's predictions on `Department` and `Attrition` <br> *Note: rounded to four (4) digits for readability* |
| **Findings**[^6] | Responding to prompted questions regarding the metrics used, activation choices made in the construction of the model `model`, and thoughts on potential ways to improve the model's performance |

[^1]: Image courtesy of the free source image site, <a href='https://www.pexels.com/photo/an-employee-feeling-the-pressure-in-the-office-7163380/' title='Link to Pexels listing for image'>Pexels</a>

[^2]: Markdown cells for instructions not annotated

[^3]: Denotes cells with completed code provided, no student coding contained

[^4]: Denotes cells added by student during scaling and encoding process, per instructions to add cells if needed

[^5]: Denotes cell added by student for easier comparison of metrics

[^6]: Markdown cells for student responses