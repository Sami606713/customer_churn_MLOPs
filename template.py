import os
import logging
logging.basicConfig(level=logging.INFO)

# make directoris
dir_=[
    os.path.join("Data","raw"),
    os.path.join("Data","process"),
    os.path.join("src","components"),
    os.path.join("src","pipeline"),
    "models",
    "notebooks",
    "reports"
]

for folder in dir_:
    if not os.path.exists(folder):
        os.makedirs(folder,exist_ok=True)

        # Make a git keep file
        git_keep=os.path.join(folder,".gitkeep")

        with open(git_keep,"w") as f:
            pass


files=[
    os.path.join("src","__init__.py"),
    os.path.join("src","utils.py"),

    os.path.join("src/components","__init__.py"),
    os.path.join("src/components","data_ingestion.py"),
    os.path.join("src/components","data_transformation.py"),
    os.path.join("src/components","model_training.py"),
    os.path.join("src/components","model_monetring.py"),
    
    os.path.join("src/pipeline","__init__.py"),
    os.path.join("src/pipeline","training_pipeline.py"),
    os.path.join("src/pipeline","prediction_pipeline.py"),
    "setup.py",
    ".env",
    "Dockerfile",
    ".dockerignore",
    "app.py",
    "requirements.txt",
    "test_environment.py"
]

for file in files:
    if(not os.path.exists(file)) or (os.path.getsize(file)==0):
        with open(file,"w") as f:
            pass
    else:
        logging.info(f"{file} already exists")