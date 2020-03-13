cd ..
MODEL=search_for_low_resource
OUTPUT_DIR=outputs/$MODEL
DATA_DIR=data/LJSpeech-1.1/processed_low_resource_1k5

export PYTHONPATH=.:$PYTHONPATH

sudo apt-get update
sudo apt-get install libsndfile1 -y

mkdir -p $OUTPUT_DIR

python train_search_seminas.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --max_epochs=10 \
  | tee -a $OUTPUT_DIR/train.log
