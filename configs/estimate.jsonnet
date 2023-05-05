local n_files = 10000;
local max_rows = 128;
local d_model = 768;
local max_len = 128;
local encoding_dim = 128;
local hidden_dim = 32;
local num_workers=100;
local n_epochs = 20;
local batch_size = 6;

{
  data_module: {
  data_path: "data/estimate/",
  n_files : n_files,
  max_rows : max_rows,
  max_len :max_len,
  batch_size : batch_size,
  num_workers : num_workers,
  shuffle : true,
  },
  vocabulary: {
      path :"vocabulary/gittables/tokens.txt",
  },
  model: {
    d_model :d_model,
    max_len : max_len,
    max_rows : max_rows,
    n_layers :6,
    n_heads :6,
    n_segments :2,
    d_k :d_model, d_v :64, d_ff :d_model*4,
    encoding_dim :encoding_dim,
    hidden_dim: hidden_dim,
    load_path: "weights/magritte_ae.pth",
    save_path : "weights/magritte_estimate.pth",
    optimizer_lr :3e-4,
  },
  trainer:{
    accelerator: "gpu",
    devices:[0],
    min_epochs:1,
    strategy:"deepspeed_stage_2",
    max_epochs:n_epochs,
    precision:"16-mixed",
    accumulate_grad_batches:1,
  },
  logger:{
     save_dir:"results/estimate/tensorboard/",
  },
    callbacks: {
    pretrain_loader:{"pretrained_path":"weights/magritte_ae.pth"},
    early_stopping:{"monitor":"val_loss", "patience":3},
    },
}