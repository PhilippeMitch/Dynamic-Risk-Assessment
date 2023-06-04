
# Project Steps Overview

This project is an example of how to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of multiple companies with thousands of clients. If the model you create and deploy is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue. The industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. Because of this, it is required to set up regular monitoring of the model to ensure that it remains accurate and up-to-date.

This project was completed by proceeding through 5 steps:

1. **Data ingestion**:<br>
   - Automatically check a database for new data that can be used for model training.
   - Compile all training data to a training dataset and save it to persistent storage.
   - Write metrics related to the completed data ingestion tasks to persistent storage.
2. **Training, scoring, and deploying**:<br>
   - Write scripts that train an ML model that predicts attrition risk, and score the model.
   - Write the model and the scoring metrics to persistent storage.
3. **Diagnostics**:
   - Determine and save summary statistics related to a dataset.
   - Time the performance of model training and scoring scripts.
   - Check for dependency changes and package updates.
4. **Reporting**:
   - Automatically generate plots and documents that report on model metrics.
   - Provide an API endpoint that can return model predictions and metrics.
5. **Process Automation**:
   - Create a script and cron job that automatically run all previous steps at regular intervals.
