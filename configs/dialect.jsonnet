local max_rows = 128;
local max_len = 128;
local seed = 42;
local n_files = 20000;
local num_workers=100;
local train_dataset = "train_augmented";
local dev_dataset = "dev_augmented";
local d_model = 768;
local max_len = std.parseInt(std.extVar('max_len'));
local encoding_dim = std.parseInt(std.extVar('encoding_dim'));
local pre_experiment =  "rowlen_"+max_len+"encoding_dim"+encoding_dim+"_gittables";
local experiment = pre_experiment;

{
  data_module: {
  train_data_path: "data/dialect_detection/"+train_dataset,
  val_data_path : "data/dialect_detection/"+dev_dataset,
  "batch_size" : 8,
  "num_workers": num_workers,
  "max_rows": max_rows,
  "n_files":n_files,
  "max_len":max_len,
  },
 vocabulary: {
      directory:"vocabulary/gittables",
  },
   model: {
    d_model:d_model,
    max_len: max_len,
    max_rows: max_rows,
    n_layers:6,
    n_heads:6,
    n_classes:4,
    d_k:d_model, d_v:64, d_ff:d_model*4,
    n_segments:2,
    encoding_dim:encoding_dim,
    # C D Q E
    weights: [0.01, 1, 0.1, 1],
    optimizer_lr:1e-5,
    save_path : "weights/magritte_dialect.pth",
  },
    trainer:{
   default_root_dir: "results/dialect/",
    accelerator: "gpu",
    devices:[0],
    min_epochs:1,
    max_epochs:20,
    precision:"16-mixed",
    log_every_n_steps:10,
    },
    logger:{
     save_dir:"results/dialect/tensorboard/",
    },
    callbacks: {
    save_magritte_model:{},
    pretrain_loader:{"pretrained_path":"weights/magritte_ae.pth"},
    early_stopping:{"monitor":"val_loss", "patience":5},
    },
}
