local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "allenai/scibert_scivocab_cased",
  cuda_device: -1,
  data_paths: {
    train: "data/mechanic/coarse/train.json",
    validation: "data/mechanic/coarse/dev.json",
    test: "data/mechanic/coarse/test.json",
  },
  loss_weights: {
    ner: 1.0,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "relation"
}

