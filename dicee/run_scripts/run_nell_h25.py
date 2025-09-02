from dicee.executer import Execute
from dicee.config import Namespace
import os
import itertools
import torch
import multiprocessing

def run_model(model, dataset_path, embed_dim, learning_rate, batch_size, nhead, num_layers, layer_mode):

    args = Namespace()

    args.byte_pair_encoding = True
    args.tokenizer_path = custom_tokenizer_path
    args.model = model
    args.dataset_dir = dataset_path
    args.embedding_dim = embed_dim
    args.lr = learning_rate
    args.batch_size = batch_size
    args.use_attention_layer = (layer_mode == "attention")
    args.use_transformer_layer = (layer_mode == "transformer")
    # args.trainer = "torchDDP"


    if args.use_transformer_layer:
        args.transformer_nhead = nhead
        args.transformer_num_layers = num_layers
    else:
        args.transformer_nhead = None
        args.transformer_num_layers = None

    if custom_tokenizer_path and os.path.isfile(custom_tokenizer_path):
        args.tokenizer_path = custom_tokenizer_path
    else:
        args.tokenizer_path = None

    args.num_epochs = 500
    args.scoring_technique = "KvsAll"
    args.training_technique= "KvsAll"
    reports = Execute(args).start()
    print("Train MRR:", reports["Train"]["MRR"])
    print("Test  MRR:", reports["Test"]["MRR"])

    with open("demofile.txt", "a") as f:
        f.write(f"Train MRR:, {reports['Train']['MRR']}\n")
        f.write(f"Test  MRR:, {reports['Test']['MRR']}\n")
        f.write(str(reports) + "\n\n")

    return reports

if __name__ == "__main__":
    multiprocessing.freeze_support()

    custom_tokenizer_path = r"..\Tokenizer\NELL_995_h25_Tokenizer_Path\tokenizer.json"
    model = ["DistMult", "ComplEx", "QMult", "Keci"]
    dataset_path = ["..\\KGs\\NELL-995-h25"]
    embed_dim = [32, 64]
    learning_rate = [0.1, 0.01, 0.001]
    batch_size = [512]     
    nhead = [2]          
    num_layers = [4]     
    # layer_modes = ["none", "attention", "transformer"]
    layer_modes = ["attention"]

    results = []
    eval_dir = "Final_Results_NELL_H25"
    os.makedirs(eval_dir, exist_ok=True)

    with open("grid_search_results_NELL_H25.txt", "w") as log:
        log.write("model,dataset,lr,embed_dim,batch_size,nhead,num_layers,layer_mode,train_mrr,test_mrr\n")

        for lm in layer_modes:
            if lm == "transformer":
                for md, ds, lr, dim, bs, nh, nl in itertools.product(
                    model, dataset_path, learning_rate, embed_dim, batch_size, nhead, num_layers
                ):
                    print(f"Running: model={md}, dataset={ds}, lr={lr}, dim={dim}, bs={bs}, nhead={nh}, layers={nl}, mode={lm}")
                    reports = run_model(md, ds, dim, lr, bs, nh, nl, lm)

                    tr = reports["Train"]["MRR"]
                    te = reports["Test"]["MRR"]
                    log.write(f"{md},{ds},{lr},{dim},{bs},{nh},{nl},{lm},{tr},{te}\n")

                    idx = lambda lst, x: lst.index(x) + 1
                    fname = (
                        f"md{idx(model,md)}"
                        f"ds{idx(dataset_path,ds)}"
                        f"lr{idx(learning_rate,lr)}"
                        f"dim{idx(embed_dim,dim)}"
                        f"bs{idx(batch_size,bs)}"
                        f"_{lm}.txt"
                    )
                    with open(os.path.join(eval_dir, fname), "w") as f_run:
                        f_run.write(f"Model: {md}\n")
                        f_run.write(f"Dataset: {ds}\n")
                        f_run.write(f"Learning Rate: {lr}\n")
                        f_run.write(f"Embed Dim: {dim}\n")
                        f_run.write(f"Batch Size: {bs}\n")
                        f_run.write(f"nhead: {nh}\n")
                        f_run.write(f"num_layers: {nl}\n\n")
                        f_run.write(f"layer_mode: {lm}\n\n")
                        f_run.write(f"Train MRR: {tr}\n")
                        f_run.write(f"Test  MRR: {te}\n")

                    results.append({
                        "model": md, "dataset": ds, "lr": lr, "embed_dim": dim, "batch_size": bs,
                        "nhead": nh, "num_layers": nl, "train_mrr": tr, "test_mrr": te,
                    })
            else:
                for md, ds, lr, dim, bs in itertools.product(
                    model, dataset_path, learning_rate, embed_dim, batch_size
                ):
                    print(f"Running: model={md}, dataset={ds}, lr={lr}, dim={dim}, bs={bs}, mode={lm}")
                    reports = run_model(md, ds, dim, lr, bs, None, None, lm)

                    tr = reports["Train"]["MRR"]
                    te = reports["Test"]["MRR"]
                    log.write(f"{md},{ds},{lr},{dim},{bs},NA,NA,{lm},{tr},{te}\n")

                    idx = lambda lst, x: lst.index(x) + 1
                    fname = (
                        f"md{idx(model,md)}"
                        f"ds{idx(dataset_path,ds)}"
                        f"lr{idx(learning_rate,lr)}"
                        f"dim{idx(embed_dim,dim)}"
                        f"bs{idx(batch_size,bs)}"
                        f"_{lm}.txt"
                    )
                    with open(os.path.join(eval_dir, fname), "w") as f_run:
                        f_run.write(f"Model: {md}\n")
                        f_run.write(f"Dataset: {ds}\n")
                        f_run.write(f"Learning Rate: {lr}\n")
                        f_run.write(f"Embed Dim: {dim}\n")
                        f_run.write(f"Batch Size: {bs}\n")
                        f_run.write(f"nhead: NA\n")
                        f_run.write(f"num_layers: NA\n\n")
                        f_run.write(f"layer_mode: {lm}\n\n")
                        f_run.write(f"Train MRR: {tr}\n")
                        f_run.write(f"Test  MRR: {te}\n")

                    results.append({
                        "model": md, "dataset": ds, "lr": lr, "embed_dim": dim, "batch_size": bs,
                        "nhead": None, "num_layers": None, "train_mrr": tr, "test_mrr": te,
                    })