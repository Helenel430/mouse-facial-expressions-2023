
mlflow experiments create -n baseline-v1.X

mlflow run . -e train --experiment-name baseline-v1.X -P dataset_version="1.1"
mlflow run . -e train --experiment-name baseline-v1.X -P dataset_version="1.2"
mlflow run . -e train --experiment-name baseline-v1.X -P dataset_version="1.3"
mlflow run . -e train --experiment-name baseline-v1.X -P dataset_version="1.4"
mlflow run . -e train --experiment-name baseline-v1.X -P dataset_version="1.5"
mlflow run . -e train --experiment-name baseline-v1.X -P dataset_version="1.6"
mlflow run . -e train --experiment-name baseline-v1.X -P dataset_version="1.7"




