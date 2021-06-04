for seed in {1337..1346}
do
  python bert.py ../data/train.csv clinicalbert --criterion auc_macro --n_context_sentences 2 --seed $seed --run_test
done
