source ./eval_func.sh


eval_cmd()
{
  echo $1 $2 $3
  eval $1 $2   us8k ~/dataset/urbansound8k/UrbanSound8K $3 $4 $5
}

 eval_cmd  ${n_last_blocks}  ${batch_size}  $1   2e-3 teacher
