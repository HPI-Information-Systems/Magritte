local n_pairs = 10000000;
local max_len = 32;
local dataset = 'gittables';
local num_workers = 100;
local d_model = 768;
local n_epochs = 15;
local pretrain_step = 0;

{
  data_module: {
  train_data_path: "data/"+dataset+"/pretrain_rowpair",
  val_data_path: "data/"+dataset+"/predev_rowpair",
  batch_size : 64,
  num_workers : num_workers,
  n_pairs : n_pairs,
  max_len :max_len,
  max_rows : 2500,
  max_percent :0.15,
  mask_special_only :true,
  positive_ratio : 0.5,
  tmp_path : "experiments/tmp/" ,
  shuffle : true,
  seed :42,
  },
  vocabulary: {
      path :"vocabulary/gittables/tokens.txt",
  },
  model: {
    d_model :d_model,
    max_len : max_len,
    n_layers :6,
    n_heads :6,
    d_k :d_model, d_v :64, d_ff :d_model*4,
    n_segments :2,
    save_path : "out/magritte_rowpair_"+pretrain_step+".pth",
    dropoout :0.1,
    optimizer_lr :0.0001,
    optimizer_warmup_steps :10000,
  },
    trainer:{
    accelerator: "gpu",
    devices:[0],
    min_epochs:1,
    max_epochs:n_epochs,
    precision:"16-mixed",
    accumulate_grad_batches:1,
    },
    logger:{
     save_dir:"experiments/pretrain_rowpair_"+pretrain_step+"/tensorboard/",
    },
    callbacks: {
    save_magritte_model:{},}
}