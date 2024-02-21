# Chat_Toxicity_NLP
An NLP project for identifying toxic language 


### Data
Potential data sources:
- https://www.kaggle.com/datasets/nkitgupta/jigsaw-regression-based-data/data
- https://www.kaggle.com/datasets/julian3833/jigsaw-unintended-bias-in-toxicity-classification/data
- https://huggingface.co/datasets/Intuit-GenSRF/combined_toxicity_profanity_v2_train_eval

### Sprints 

1. decide on the entire bias dataset or tag a subset of 500k. The reason I'm leaning bias despite hugging being more model-ready is we have more documentation on the bias dataset through the competition description, and preprocessing would be a skill they'd like to see. 
2. a)Sprint 1 would be to clean, preprocess, and combine our eda into one document
2. b) We only care about toxicity at this point, tags are extra, and other metrics such as obscene, threat, etc are also extra
3. a) Sprint 2 would be to get our base model, likely logistic regression 
3. b) we then improved on our base model and get a toxicity prediction result we're satisfied with
4. Sprint 3 is to pipeline and automate this process, have a deployable and replicable ready-to-use model on hugging face with the use of docker, containerization, and other ML methods mentioned during our industry mentor meeting 
