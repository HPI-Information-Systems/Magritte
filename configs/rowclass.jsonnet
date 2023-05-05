local n_files = 50000;
local max_rows = 128;
local dataset = 'gittables';
local d_model = 768;
local encoding_dim = 128;
local max_len = 128;
local num_workers = 100;
local validation_dataset = std.extVar('validation_dataset');

{
  data_module: {
   data_path: "data/row_classification/strudel_annotations.jsonl",
   shuffle:true,
   batch_size : 8,
   num_workers: num_workers,
   max_rows: max_rows,
   max_len: max_len,
   n_files: n_files,
   train_datasets: ["deex","saus","cius","govuk"],
   val_dataset_name: validation_dataset,
  },
  vocabulary: {
      type:"from_files",
      directory:"vocabulary/"+dataset,
  },
  model: {
    d_model:d_model,
    max_len: max_len,
    max_rows: max_rows,
    n_layers:6,
    n_heads:6,
    n_classes:6,
    d_k:d_model, d_v:64, d_ff:d_model*4,
    n_segments:2,
    encoding_dim:encoding_dim,
    ignore_class:"empty",
    classes_weights: [0.01,1, 1, 1, 1, 1],
    optimizer_lr:1e-4,
    save_path:"weights/rowclass/magritte_lineclass"+validation_dataset+".pth"
  },
    trainer:{
    accelerator: "gpu",
    devices:[0],
    min_epochs:1,
    max_epochs:20,
    precision:"16-mixed",
    log_every_n_steps:10,
    },
    logger:{
     save_dir:"results/rowclass/tensorboard/",
    },
    callbacks: {
    pretrain_loader:{"pretrained_path":"weights/magritte_ae.pth"},
    early_stopping:{"monitor":"val_loss", "patience":5},
    },
}