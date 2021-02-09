set -e
set -x

virtualenv -p python3
source ./bin/activate

chmod +x active_sensing
chmod +x api
chmod +x automl
chmod +x datasets
chmod +x docs
chmod +x evaluation
chmod +x feature_selection
chmod +x imputation
chmod +x interpretation
chmod +x prediction
chmod +x preprocessing
chmod +x tmp
chmod +x treatments
chmod +x tutorial
chmod +x uncertainty
chmod +x utils

pip install tensorflow
pip install -r requirements.txt
python -m ./api/main_api_prediction