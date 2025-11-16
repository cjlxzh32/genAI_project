export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# # Generate a cat using SD3.5 Large model (at models/sd3.5_large.safetensors) with its default settings
# python3 sd3_infer.py --prompt "cute wallpaper art of a cat" > "result/log.log" 2>&1
# # Or use a text file with a list of prompts, using SD3.5 Large
# python3 sd3_infer.py --prompt path/to/my_prompts.txt --model models/sd3.5_large.safetensors
# # Generate from prompt file using SD3.5 Large Turbo with its default settings
# python3 sd3_infer.py --prompt path/to/my_prompts.txt --model models/sd3.5_large_turbo.safetensors
# # Generate from prompt file using SD3.5 Medium with its default settings, at 2k resolution
# python3 sd3_infer.py --prompt path/to/my_prompts.txt --model models/sd3.5_medium.safetensors --width 1920 --height 1080
# # Generate from prompt file using SD3 Medium with its default settings
# python3 sd3_infer.py --prompt path/to/my_prompts.txt --model models/sd3_medium.safetensors

# python3 test.py > "result/log.log" 2>&1
NUM_SHARDS=8

STEPS=40
CFG=6
SEED=42
WIDTH=1024
HEIGHT=1024

for RANK in $(seq 0 $((NUM_SHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$RANK \
  python3 sd3_infer.py \
    --steps $STEPS \
    --cfg $CFG \
    --seed $SEED \
    --width $WIDTH  \
    --height $HEIGHT    \
    --num_shards $NUM_SHARDS \
    --shard_index $RANK \
    --device cuda:0 \
    --text_encoder_device cuda \
    > "result/shard_${RANK}.log" 2>&1 &
done

