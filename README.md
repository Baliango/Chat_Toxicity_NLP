# Project Overview

The goal with this project is to train an NLP (Natural Language Processing) model to recognize toxicity in comment text, and then containerize and deploy said model into an online API for easy use.  To accomplish this we have selected several toxicity datasets from related Kaggle competitions and Hugging Face libraries for training purposes.  We will begin training the model on one such dataset and potentially update and improve it on others as the project proceeds.


## Data

We have filtered out two potential sources of data for our training purposes.  

Our first dataset comes from a competition for unintended bias classification for Kaggle, the link to the dataset and all pertinent related information can be found [here.](https://www.kaggle.com/datasets/julian3833/jigsaw-unintended-bias-in-toxicity-classification/data)

From our EDA and reading through the documentation on the dataset we've ascertained that these datasets were created through direct human rating in which live individuals were paid to rate a comments toxicity on a float scale of 0-1, taking into consideration factors such as: toxicity, severe toxicity, obscene, insult, or threat.  Additionally, for a small subset of the data extra factors were provided which allow for identification metrics such as race, sex, and religion among others.  

Though the datasets show evidence of manipulation presumably by the original poster the key dataframes for building a new model are still present and untouched.  The useful ones for our purposes are as follows:

**Train** - 
 
| id   | target | comment_text                                                                                                           | severe_toxicity | obscene | identity_attack | insult | threat | asian | atheist | bisexual | black | buddhist | christian | female | heterosexual | hindu | homosexual_gay_or_lesbian | intellectual_or_learning_disability | jewish | latino | male | muslim | other_disability | other_gender | other_race_or_ethnicity | other_religion | other_sexual_orientation | physical_disability | psychiatric_or_mental_illness | transgender | white | created_date                   | publication_id | parent_id | article_id | rating   | funny | wow | sad | likes | disagree | sexual_explicit | identity_annotator_count | toxicity_annotator_count |
|------|--------|------------------------------------------------------------------------------------------------------------------------|----------------|---------|-----------------|--------|--------|-------|---------|----------|-------|----------|-----------|--------|--------------|-------|--------------------------|--------------------------------------|--------|--------|------|--------|-----------------|--------------|-------------------------|---------------|------------------------|---------------------|------------------------------|-------------|-------|--------------------------------|---------------|-----------|------------|----------|-------|-----|-----|-------|----------|-----------------|-------------------------|--------------------------|
| 59848| 0.0    | "This is so cool. It's like, 'would you want your mother to read this??' Really great idea, well done!"                | 0.0            | 0.0     | 0.0             | 0.0    | 0.0    | N/A   | N/A     | N/A      | N/A   | N/A      | N/A       | N/A    | N/A          | N/A   | N/A                      | N/A                                  | N/A    | N/A    | N/A  | N/A    | N/A             | N/A          | N/A                     | N/A           | N/A                    | N/A                 | N/A                          | N/A         | N/A   | 2015-09-29 10:50:41.987077+00 | 2             | N/A       | 2006       | rejected | 0     | 0   | 0   | 0     | 0        | 0.0             | 0                       | 4                        |
| 59849| 0.0    | "Thank you!! This would make my life a lot less anxiety-inducing. Keep it up, and don't let anyone get in your way!" | 0.0            | 0.0     | 0.0             | 0.0    | 0.0    | N/A   | N/A     | N/A      | N/A   | N/A      | N/A       | N/A    | N/A          | N/A   | N/A                      | N/A                                  | N/A    | N/A    | N/A  | N/A    | N/A             | N/A          | N/A                     | N/A           | N/A                    | N/A                 | N/A                          | N/A         | N/A   | 2015-09-29 10:50:42.870083+00 | 2             | N/A       | 2006       | rejected | 0     | 0   | 0   | 0     | 0        | 0.0             | 0                       | 4                        |

This is the primary dataset for training our model.  The two key columns are our comment_text and target columns, however, in future iterations of our model there are additional options available for multilabel classification which could potentially be used to refine our models predictions and output.  

The N/A labeled columns are due to the previously mentioned smaller subset of identity labeled comment texts that were additionally included in this dataframe.  Looking at their column titles reveals that there is some interesting information available here which might argue a position of using the smaller subset for specific identification model training.  

**Test** -

| id      | comment_text                                                                        |
|---------|-------------------------------------------------------------------------------------|
| 7097320 | "[ Integrity means that you pay your debts.] Does this apply to President Trump too?" |
| 7097321 | "This is malfeasance by the Administrator and the Board. They are wasting our money!"  |

A dataset with unlabeled target classes and blank comment texts for model testing purposes.  There is additionally two expanded versions which include the additional columns available from the train dataset.

**Identification Dataframes** - 

Finally, there are two dataframes which specifically pertain to the subset of data in train that are labeled with extra identification and toxicity metrics.  

**Toxicity_individual_annotations** - 

| id   | worker | toxic | severe_toxic | identity_attack | insult | obscene | sexual_explicit | threat |
|------|--------|-------|--------------|-----------------|--------|---------|-----------------|--------|
| 59859| 0      | 1     | 0            | 0               | 1      | 1       | 0               | 0      |
| 59859| 1      | 1     | 0            | 0               | 0      | 1       | 0               | 0      |

A basic boolean dataframe where a worker reads a comment and determines whether the content qualifies for a specific label.  As multiple workers review multiple comments the boolean values devolve into float values for the main train dataset.  

**Identity_individual_annotations** -

| id   | worker | disability                     | gender | race_or_ethnicity | religion | sexual_orientation |
|------|--------|-------------------------------|--------|-------------------|----------|--------------------|
| 59856| 211    | none                          | none   | none              | none     | none               |
| 59856| 683    | intellectual_or_learning| none   | none              | none     | none               |

Categorical values for the subset of data that includes indentification metrics.  It is unclear in our EDA and in the original documentation whether a category determines whether a target is simply mentioned in a comment or if it is specifically used in a negative light.  Should we pursue multilabel classification with identification factors this readme will be updated with more information accordingly.    


------------------------


The secondary dataset was sourced from Hugging Face, specifically from the Intuit-GenSRF organization which features optimized datasets in multiple languages designed for training models for content moderation.  The link to the specific dataset we selected for our purposes can be accessed [here.](https://huggingface.co/datasets/Intuit-GenSRF/combined_toxicity_profanity_v2_train_eval)  Additional NLP oriented datasets from the organization can additionally be accessed on their Hugging Face profile at this [link.](https://huggingface.co/Intuit-GenSRF)

**Train/Validation** - 

|    | text                                          | labels               | encoded_labels       |
|----|-----------------------------------------------|----------------------|----------------------|
| 0  | Football positions \n\nHi Mehusdon. When you'... | []                   | [0, 0, 0, 0, 0, 0, 0, 0, 0] |
| 1  | Thank you SO MUCH for your Nazi-like oversight... | [toxic, profane, insult] | [1, 1, 0, 0, 0, 0, 0, 0, 0] |
| 2  | Darkwind \n Stick it up your arse you offensi... | [toxic, profane]     | [1, 1, 0, 0, 0, 0, 0, 0, 0] |

The files included are 6 train sets and 1 validation set all following identical format as Parquet files.  They are intended for multilabel classification and as such feature 9 different toxicity labels that are already pre-encoded.  Though we have decided to move forward with the Kaggle dataset for learning purposes and further optimization and customization we are keeping the Hugging Face dataset on the backburner for reference and potential use in the future.


## Sprint Information (Project Goals/Updates)

- [x] Initialize GitHub Repository, select dataset(s) for review and create preliminary ReadMe
- [x] Preliminary EDA on data, including cleaning and noting any problem areas
- [x] Preliminary model building, determine strengths and weaknesses of certain approaches 
- [x] Advanced model research and prototyping 
- [x] Select final model: distilbert-base-uncased
- [x] Optimize distilbert parameters (Completed for now, might revert to multilabel classification)
- [x] Build base pipeline system to test models performance 
- [x] Package model and scripts into Docker container
- [ ] Research hosting platforms, determine deployment goals for final product
- [ ] Revisit model for futher iteration and improvement 
