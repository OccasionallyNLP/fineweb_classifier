accelerate launch --num_processes=8 inference_latest.py --batch_size 1024 --test_data /home/work/g-earth-22/hoyoun/dclm_baseline_1.0_train_000/dclm_baseline_1.0_train_00.jsonl --output_dir dclm_baseline_1.0_train_00

# accelerate launch --num_processes=8 inference.py --batch_size 1024 --test_data /home/work/g-earth-22/hoyoun/dclm_baseline_1.0_train_000_1.jsonl --output_dir dclm_baseline_1.0_train_0 
