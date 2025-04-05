for L in 1 2 3
do
    for d in 64 128 256
    do
        echo "============== Running model scaling experiments for LSTM with (L, d) = ($L, $d) =============="
        python run_exp.py --p 31 --operator + --num_layers $L --embedding_size $d --hidden_size $d --model lstm --optimizer adamw --log_dir /network/scratch/d/dhruv.sreenivas/ift-6135/hw2/logs/lstm/scale_model/layers-$L-dim-$d --exp_name scale_model_layers_${L}_dim_${d} --multiple
    done
done

for L in 1 2 3
do
    for d in 64 128 256
    do
        echo "============== Running model scaling experiments for GPT with (L, d) = ($L, $d) =============="
        python run_exp.py --p 31 --operator + --num_layers $L --embedding_size $d --hidden_size $d --model gpt --optimizer adamw --log_dir /network/scratch/d/dhruv.sreenivas/ift-6135/hw2/logs/gpt/scale_model/layers-$L-dim-$d --exp_name scale_model_layers_${L}_dim_${d} --multiple
    done
done