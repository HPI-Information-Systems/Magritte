local n_files = 1000000;
local max_rows = 128;
local dataset = 'gittables';
local d_model = 768;
local max_len = 128;
local encoding_dim = 128;
local num_workers=100;
local n_epochs = 10;
local batch_size = 64;

{
  data_module: {
  data_path: "data/gittables/csv",
  save_dir:"results/pretrain_ae/",
  n_files : n_files,
  max_rows : max_rows,
  max_len :max_len,
  batch_size : batch_size,
  num_workers : num_workers,
  shuffle : true,
  train_val_split : 0.9,
  },
  vocabulary: {
      path :"vocabulary/gittables/tokens.txt",
  },
  model: {
    d_model :d_model,
    max_len : max_len,
    max_rows : max_rows,
    n_layers :6, #6
    n_heads :6,
    n_segments :2,
    d_k :d_model, d_v :64, d_ff :d_model*4,
    encoding_dim :encoding_dim,
    beta : 1.5,
    no_grad: ["embedding.*","norm.*","encoding_layers.*"],
    load_path: "out/magritte_rowpair_1.pth",
    save_path : "out/magritte_ae.pth",
    dropoout :0.1,
    optimizer_lr :10e-4,
  },
    trainer:{
    accelerator: "gpu",
    strategy:"deepspeed_stage_2",
    devices:[0],
    min_epochs:1,
    max_epochs:n_epochs,
    precision:"16-mixed",
    accumulate_grad_batches:4,
    },
    logger:{
     save_dir:"results/pretrain_ae/tensorboard/",
    },
    callbacks: {
    pretrain_loader:{pretrained_path:"out/magritte_rowpair_1.pth",},
    early_stopping:{"monitor":"val_loss", "patience":3},
    }
}