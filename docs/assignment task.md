Machine Learning Task for TDT4173
Append Consulting AS x Hydro ASA
Task Overview
This task is part of a consultancy project conducted by Append Consulting for Hydro.
Hydro is a Norwegian industrial company with operations in energy, aluminum, and
recycling. Append is a consultancy firm mainly consisting of students from NTNU, that
works with data science, artificial intelligence, and system development to help companies
make better use of data and technology in their operations.
The aim of this task is to develop accurate - but conservative - forecasts of incoming raw
material deliveries, to be used in a larger optimization tool developed by Append.
Task Description
You are provided historical data on raw material deliveries and orders through the end
of 2024. Each raw material is identified by a unique rm id. The goal is to develop a
model that forecasts the cumulative weight of incoming deliveries of each raw material
from January 1, 2025, up to any specified end date between January 1 and May 31, 2025.
For any end date within this range, the model should predict the total weight in kg of
a raw material delivered from and including January 1, 2025, to and including the end
date.
Dataset Overview
The datasets are organized as follows:
• data/kernel/receivals.csv: The primary dataset containing historical records of
material receivals. Each entry includes a timestamp, the quantity received, and the
corresponding rm id.
• data/kernel/purchase orders.csv Contains information on ordered quantities
and expected deliveries.
• data/extended/materials.csv (Optional): Metadata on various raw materials,
including categories and classifications.
1
• data/extended/transportation.csv (Optional): Transportation-related data that
could affect delivery times and consistency.
Evaluation
Quantile Error at 0.2 (Asymmetric Loss)
Let there be N raw materials indexed by i = 1, . . . , N . Over a forecasting window of h
days, define
Ai =
hX
t=1
yi,T +t, Fi =
hX
t=1
ˆyi,T +t,
as the actual and forecasted total deliveries for material i, respectively.
To evaluate performance, we compute the quantile loss at the 0.2 level:
QuantileLoss0.2(Fi, Ai) = max (0.2 · (Ai − Fi), 0.8 · (Fi − Ai)) .
The overall metric is the average quantile loss across all materials:
QuantileError0.2 = 1
N
NX
i=1
QuantileLoss0.2(Fi, Ai).
This metric penalizes overestimation more than underestimation, which aligns with the
practical needs of smelting. If we underestimate the available materials, the smelt can
usually continue with what is on hand. However, if we overestimate, we risk planning a
smelt that cannot be completed due to missing resources. Therefore, it’s better for the
model to be slightly cautious and predict too little rather than too much.

TDT4173 Modern Machine Learning in Practice
Course Project
September 15, 2025
1 Submission
• Remember to register your exam!
• Each group submit three things:
– Select 2 predictions on Kaggle
– Two short Jupyter notebooks (Short notebook 1.ipynb and Short notebook 2.ipynb)
* They contain only the necessary steps to reproduce your two selected Kaggle predictions, re-
spectively.
* We may re-run your notebook to check whether the results are reproducible. At least one of
the two short notebooks should repeat your best Kaggle score.
– A report.
* You can use a PDF file (Report.pdf) or a long Jupyter notebook (Report.ipynb)
* The report summarizes all steps in your group work e.g., exploratory data analysis, feature
engineering, all models/algorithms no matter they are helpful or not.
* Remember to include model interpretation in the report.
* We will not run your long notebook. So, if you use a Jupyter notebook as the report, you should
include the running result, not only the source code.
* There is no minimum or maximum length limit to the report.
• You should submit
– the predictions to Kaggle
– the notebooks and report to Blackboard
– Please include your group number in your Kaggle team name, e.g., “[34] A Fantastic Team”
* No group number and duplicated group numbers are treated as hacking
• You can use whatever programming languages, tools, platforms, software libraries, or file formats you
want during development, but you should use Jupyter notebooks for your submission.
• Do not use separate files. Put all code in the Jupyter notebooks.
• The project deadline is November 9, 2025 at 22:00 Oslo time.
– The deadline is strict. The competition will be closed immediately after the submission deadline.
Being late in any part for even a few seconds in Blackboard is treated as a late submission.
– If we cannot find your predictions, notebook(s), or report after the deadline, we will ask you to
submit them again and treat them as a late submission.
– Stop submitting after the deadline. We grade your last attempt. If your last attempt is after the
deadline, your team will have a late submission.
1
• Late submission deadline is November 13, 2025 at 14:00 Oslo time.
– A late submission is not recommended. It will cause a -30 deduction in your course points.
– If you submit even later (i.e., after the late submission deadline), the whole group fail the course.
– Late predictions will be submitted to another Kaggle link (given later)
• Most excuses, e.g., short-term sickness and other activities, are not accepted for a deadline extension.
Start early for the project work.
• At the beginning of each notebook and the report, write your full names, student ids, and Kaggle team
name. Student ID is the one on your student card, not the candidate number in the course. You may fail
the course if we cannot map your notebooks and report to your Kaggle predictions.
2 Grading
• Project points = base points + possible deductions
• Convert points to letter grade
– A: 89-100
– B: 77-88
– C: 65-76
– D: 53-64
– E: 41-52
– F: 0-40
• Base points
– proportional to the number of Virtual Teams (VTs) you defeat
– max 100 (defeat all VTs) and min 41 (defeat 1 VT)
– if you defeat 0 VT, you fail the course
• Virtual Teams (VTs)
– VTs are prepared by the teachers and assistants
– Each VT uses the same data as you have
– All VTs will be frozen (i.e., not changed) after 19/October/2025
• We use the private leaderboard to calculate base points
– You can see only the public leaderboard before the competition closes
– You select two predictions before the deadline (your short notebooks should be able to reproduce
the selected predictions)
– We use the best between your selected in the private leaderboard to calculate your base points
– Notice: Ranking in the public leaderboard has no effect in grading, and it is possible that there is
(big) difference between the two leaderboards.
• More about the reproducibility
– No external data is allowed.
– Your short notebooks will be run in an offline setting
– Your short notebooks should start from the given raw data
* You can include a configuration file (e.g., requirements.txt or environment.yaml) to specify the
required software packages and their versions.
2
* Before running your code, we will delete everything from you except the two short notebooks
and the configuration file.
* During the running, your program can store some middle results as disk files.
– You can use a few programming constants, but writing massive data in the code and in the configu-
ration file is not allowed.
– The maximum running time for a short notebook is 12 hours on a normal PC.
* We don’t give strict specifications of ”normal PC” because you may not find a machine with
exactly the same configuration.
* You can think a ”normal PC” is an average student’s laptop, with maximum 4 CPU cores and
32G RAM.
* If you use a GPU, please specify the model and required software. We will try to find a similar
GPU. But don’t expect a high-end GPU. Multiple GPUs are not allowed.
– You can directly specify a few hyperparameter values in the short notebooks, but you must describe
how you obtain the values in the report.
– The short notebooks should have only the relevant parts to generate your selected Kaggle predic-
tions.
– A small difference (not affecting your ranking against VTs) is acceptable. Otherwise, we will run
your notebook again and pick the best score among our multiple runs for grading.
• Possible deductions
– pass the individual assignment in the second chance (-5); the deduction applies to individuals, not
the whole team.
– late submission (between project deadline and late submission deadline) of the project (-30)
– no exploratory data analysis (-3). To avoid the deduction, you should do at least four or more items
of the following list:
* Search domain knowledge
* Check if the data is intuitive
* Understand how the data was generated
* Explore individual features
* Explore pairs and groups of features
* Clean up features
– only one type of predictor is used (-3). To avoid the deduction, you should show that you have tried
two or more types of predictors in the report (e.g., XGBoost and Random Forest). It is allowed to
use only one type of predictor in a short notebook for Kaggle predictions.
– no feature engineering (-3). To avoid the deduction, you should show that you have tried one or
more feature engineering techniques (i.e., feature selection and/or feature extraction) in the report.
– no model interpretation (-3). To avoid the deduction, you should show one or more model interpre-
tation results (e.g., PDP, feature importance, LIME) in the report.
– Please use clear section titles in the report for easy lookup of the above parts.
• All deductions are binary. That is, there are only two possibilities: full deduction and no deduction.
There is no intermediate deduction.
• A grading example: a student
– passes the individual assignment in the first chance
– submits the project in time
– his or her team defeats 7 of 10 VTs
– no model interpretation in the notebooks (-3)
3
– Then the student’s course points
100− 41
* base points= 41 +
10− 1
* course points= 80− 3= 77
* The corresponding letter grade is B.
× (7− 1) ≈ 80
4