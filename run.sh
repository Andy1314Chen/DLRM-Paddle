

# train model with sample data or full data
python -u tools/trainer.py -m models/rank/dlrm/"$1".yaml

## infer model with sample data or full data
python -u tools/infer.py -m models/rank/dlrm/"$1".yaml