local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "allenai/scibert_scivocab_cased",
  cuda_device: 0,
  data_paths: {
    train: "preprocessing/label-studio_exports/output/NER_dataset_all_train.json",
    validation: "preprocessing/label-studio_exports/output/NER_dataset_all_test.json",
    test: "data/mechanic/coarse/test.json",
  },
  loss_weights: {
    ner: 0.2,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "relation",
}
