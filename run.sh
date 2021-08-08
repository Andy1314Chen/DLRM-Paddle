
echo "Start Training..."
# train model with sample data or full data
python -u tools/trainer.py -m models/rank/dlrm/"$1".yaml

rm -rf tools/utils/__pycache__ models/rank/dlrm/__pycache__

echo "Start Testing..."
## infer model with sample data or full data
python -u tools/infer.py -m models/rank/dlrm/"$1".yaml

rm -rf tools/utils/__pycache__ models/rank/dlrm/__pycache__

echo " Done~"