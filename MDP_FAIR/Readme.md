## Models
```angular2html
model 1：
davinci:ft-greendynamics:custom-model-name-2023-04-01-06-53-54
ft-bnJ3Ums4MQlh3YPgCfrypnRL

model 2：
davinci:ft-greendynamics:custom-model-name-2023-04-01-10-28-23
ft-izfAT1rhkcA91wuu7GR6NmJZ
```

## Execution
1. Data Processing
After getting data in a csv file with formation of `prompt, completion`, run:
```angular2html
openai tools fine_tunes.prepare_data -f <csv_data>
```
It will generate a json data with a formation fitting openai model fit in

2. Train model & get suffix name
```angular2html
openai api fine_tunes.create -t <prepared_json_file> -m davinci --suffix "custom model name"
```
3. To get the current status of model (with model id & suffix name), run:
```angular2html
openai api fine_tunes.get -i <model_id>
```
If you want to cancel the training model, run:
```angular2html
openai api fine_tunes.cancel -i <model_id>
```
4. Output the model training result
```angular2html
openai api fine_tunes.results -i <model_id> > result.csv
```