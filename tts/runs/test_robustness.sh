cd ../
OUTPUT_DIR=checkpoints/robustness
MODEL=checkpoint_best.pt
DATA_DIR=data/LJSpeech-1.1/processed_robustness

export PYTHONPATH=.:$PYTHONPATH

python test.py \
  --data=$DATA_DIR \
  --attn_constraint \
  --output_dir=$OUTPUT_DIR \
  --checkpoint_path=$OUTPUT_DIR/${MODEL} 
