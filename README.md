# MACHINE LAERNING project report

The main idea of this project is to build a simple sequence labelling system using statistical model (specifically implementing a `hidden Markov Model`) to perform a sentiment classification task

> This is the project report for the final project of `Machine Learning` class 2016 Fall at Singapore University of Technology and Design(SUTD)

> Group Member:

> - LU Jialin
> - RUAN Jiayi
> - GUAN Xianyue

## Structure of code

- `sequence_labelling_system.py`:

  - parameter calculation
  - Viterbi Algorithm
  - Top-k Viterbi Algorithm
  - State Rerank

- `eval_and_test.py`

  - evaluate script

- `test.py`

  - lines of code of running the system

- `test_module.py`

  - functions used in `test.py`

## HOW To run the code?

simple use python3 to run `test.py` in the project folder.

note that in `test.py` there mainly 9 lines of code and I will briefly introduce what they would do.

```
project_part_2_prepare()
```

this function is to test for the basic funcitional ability of the system.You can comment it if you do not like it

```
project_part_2()
```

this funcion is to run the train and predict procedure on all 4 datasets(in fact it calls 4 times `learn_and_predict_evaluate_part_2()`on different data sets)

```
learn_and_predict_evaluate_part_2("EN")
```

this function is to run the labelling system for part2 in data set "EN", you can change it to "CN" or "ES" or "SG" if you like, or you can just comment it to disable it.

```
project_part_3()
```

this funcion is to run the train and predict procedure on all 4 datasets(in fact it calls 4 times `learn_and_predict_evaluate_part_3()`on different data sets)

```
learn_and_predict_evaluate_part_3("EN")
```

this function is to run the labelling system for part4 in data set "EN", you can change it to "CN" or "ES" or "SG" if you like, or you can just comment it to disable it.

```
project_part_4()
```

this funcion is to run the train and predict procedure on all 4 datasets(in fact it calls 4 times `learn_and_predict_evaluate_part_4()`on different data sets)

```
learn_and_predict_evaluate_part_4("CN")
```

this function is to run the labelling system for part4 in data set "EN", you can change it to "CN" or "ES" or "SG" if you like, or you can just comment it to disable it.

```
project_part_5(5)
```

this funcion is to run the train and predict procedure on all 4 datasets(in fact it calls 4 times `learn_and_predict_evaluate_part_4()`on different data sets)

since we are using a top-k rerank method, we must define the k(which is specified by the `number` parameter)

```
learn_and_predict_evaluate_part_5("EN",number=5)
```

this function is to run the labelling system for part5 in data set "EN", you can change it to "CN" or "ES" or "SG" if you like, or you can just comment it to disable it.

since we are using a top-k rerank method, we must define the k(which is specified by the `number` parameter)

```
test()
```

this function is to run the evaluation on each output from part2 to part5 on all 4 data sets
