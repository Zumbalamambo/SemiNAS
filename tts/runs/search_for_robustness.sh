cd ..
MODEL=search_for_robust
OUTPUT_DIR=outputs/$MODEL
DATA_DIR=data/LJSpeech-1.1/processed_robust100

export PYTHONPATH=.:$PYTHONPATH

sudo apt-get update
sudo apt-get install libsndfile1 -y

mkdir -p $OUTPUT_DIR

python train_search_seminas.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  | tee -a $OUTPUT_DIR/train.log
