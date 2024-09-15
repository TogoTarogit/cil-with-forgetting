# CUDA_VISIBLE_DEVICES="0" python train_cvae.py --config mnist.yaml --data_path ./dataset --dataset fashion
# CUDA_VISIBLE_DEVICES="0" python train_classifier.py --data_path ./dataset --dataset fashion
# CUDA_VISIBLE_DEVICES="0" python generate_samples.py  --ckpt_folder ./results/mnist/2024_02_01_094620 --label_to_generate 0 --n_samples 1000
# CUDA_VISIBLE_DEVICES="0" python generate_samples.py --ckpt_folder ./results/mnist/2024_02_01_094620 --label_to_generate 1 --n_samples 1000
# CUDA_VISIBLE_DEVICES="0" python evaluate_with_classifier.py --sample_path ./results/mnist/2024_02_01_094620 --label_of_dropped_class 0 --dataset fashion
# CUDA_VISIBLE_DEVICES="0" python evaluate_with_classifier.py --sample_path ./results/mnist/2024_02_01_094620 --label_of_dropped_class 1 --dataset fashion

# random が動作することの確認
# ----------------------------------------------------
# 実験設定
cuda_num=0
# サンプルとして出力する画像の枚数
# n_samples=1000
n_samples=10000
dataset="mnist"
forgetting_method="random"
# yaml=$dataset".yaml"
contents_discription=""
learn=0
forget=1
# --------------------------
echo "start VAE training. no train data class is $learn"
    vae_output_str=$(
        CUDA_VISIBLE_DEVICES="$cuda_num" python train_cvae.py --remove_label $learn --data_path ./dataset --dataset $dataset
    ) 
echo "start fim calculation for ewc and no sa ewc" 
#output から save dir を抜き取る
vae_save_dir=$(echo "$vae_output_str" | grep -oP 'vae save dir:\K[^\n]*')
echo "VAE save dir is $vae_save_dir"
echo "start FIM calculation"
CUDA_VISIBLE_DEVICES="$cuda_num" python calculate_fim.py --ckpt_folder $vae_save_dir
    
sa_output_str=$(
                CUDA_VISIBLE_DEVICES="$cuda_num" python train_forget.py --ckpt_folder $vae_save_dir --label_to_drop $forget --lmbda 100 --forgetting_method $forgetting_method
            ) 
echo "sa output is $sa_output_str"
# モデルの評価を行う
# 10000枚の画像を生成
CUDA_VISIBLE_DEVICES=$cuda_num python generate_samples.py --ckpt_folder $sa_ewc_save_dir --label_to_generate $learn --n_samples $n_samples
# 分類機で精度を出す
results=$(
    CUDA_VISIBLE_DEVICES=$cuda_num python evaluate_with_classifier.py --sample_path $sa_ewc_save_dir --label_of_dropped_class $learn --dataset $dataset
    )