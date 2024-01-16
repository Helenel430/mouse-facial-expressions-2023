mlflow experiments create -n baseline-noaug-v1.X

mlflow run . -e train --experiment-name baseline-noaug-v1.X -P dataset_version="1.1" -P train_augmentation="None"
mlflow run . -e train --experiment-name baseline-noaug-v1.X -P dataset_version="1.2" -P train_augmentation="None"
mlflow run . -e train --experiment-name baseline-noaug-v1.X -P dataset_version="1.5" -P train_augmentation="None"
mlflow run . -e train --experiment-name baseline-noaug-v1.X -P dataset_version="1.7" -P train_augmentation="None"

