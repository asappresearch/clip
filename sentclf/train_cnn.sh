for seed in {1337..1346}
do
  python neural_baselines.py ../data/train.csv cnn --criterion auc_macro --vocab_file cnn_vocab.txt --embed_file cnn_embs.pkl --seed $seed --run_test
done
