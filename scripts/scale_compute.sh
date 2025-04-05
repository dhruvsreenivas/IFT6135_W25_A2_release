for bs in 32 64 128 256 512
do
    echo "============== Running compute scaling experiments for LSTM (batch size $bs) =============="
    python run_exp.py --p 31 --operator + --train_batch_size $bs --n_steps 20001 --model lstm --optimizer adamw --log_dir /network/scratch/d/dhruv.sreenivas/ift-6135/hw2/logs/lstm/scale_compute/batch-size-$bs --exp_name scale_compute_$bs --multiple
done

for bs in 32 64 128 256 512
do
    echo "============== Running compute scaling experiments for GPT (batch size $bs) =============="
    python run_exp.py --p 31 --operator + --train_batch_size $bs --n_steps 20001 --model gpt --optimizer adamw --log_dir /network/scratch/d/dhruv.sreenivas/ift-6135/hw2/logs/gpt/scale_compute/batch-size-$bs --exp_name scale_compute_$bs --multiple
done