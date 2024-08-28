# Instruction
The scripts to run the pipeline on raw files of the EPO and USPTO patents are provided. The pipeline uses existing models for reference extraction and field extraction. 
For EPO patents use ```EPO-pipeline.py``` script. For USPTO patents published between 1990-2001 use ```pipeline-uspto-v3.py```. For USPTO patents published between 2002-2004 use ```uspto-v4.py```. 
For USPTO patents published between 2005-2022 use ```pipeline-uspto-v1.py``` script. 

You need to define the following variables in these scripts:


```
WOS_PATH =  "/WOS/pub" (Path to the WOS dataset)
Source_PATH =  '/WOS/source2213.csv'  (Path to the WOS dataset)
PUB_PATH =  '/WOS/book2213.csv'  (Path to the WOS dataset)
USPTO_DATA_DIR_PATH = '/uspto/' (Path to the directory of the uspto patents)
model_checkpoint_RE = "Models/ref-extraction-complete.model" (Path to the fine-tuned reference extraction model)
saved_model_checkpoint_FE = '/Models/field-extraction-complete.model'  (Path to the fine-tuned field extraction model)
USPTO_DB_PATH = '/USPTO-v3.db'  (Path to the sqlite database to store the extracted references)
```

The scheme of the table for storing the extracted references is as follows: The name of the Table is ``````
