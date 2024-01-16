
mlflow experiments create -n sex-differences
mlflow run . -e trainv3 --experiment-name "sex-differences" -P dataset_version="3.0"

# mlflow experiments create -n "sex-differences;lr=1e-2"
# mlflow run . -e trainv3 --experiment-name "sex-differences;lr=1e-2" -P dataset_version="3.0" -P learning_rate="0.01"

# mlflow experiments create -n "sex-differences;lr=1e-3"
# mlflow run . -e trainv3 --experiment-name "sex-differences;lr=1e-3" -P dataset_version="3.0" -P learning_rate="0.001"

# mlflow experiments create -n "sex-differences;lr=1e-4"
# mlflow run . -e trainv3 --experiment-name "sex-differences;lr=1e-4" -P dataset_version="3.0" -P learning_rate="0.0001"

# mlflow experiments create -n "sex-differences;lr=1e-5"
# mlflow run . -e trainv3 --experiment-name "sex-differences;lr=1e-5" -P dataset_version="3.0" -P learning_rate="0.00001"
