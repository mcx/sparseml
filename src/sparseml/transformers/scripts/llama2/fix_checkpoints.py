import os
import torch

def fix_checkpoint(chkpt_file_path):
    dd = torch.load(chkpt_file_path)
    new_dd = {k: dd[k] for k in dd.keys() if k.find(".module") < 0}
    for k in dd.keys():
        t = k.find(".module")
        if t < 0:
            continue
        new_key = k[:t] + k[t+len(".module"):]
        new_dd[new_key] = dd[k]
    torch.save(new_dd, f"{ckpt_file_path}.new")


model_dir = "/network/tuan/models/llama/GSM8K/ongoing/llama-2-7b_pruned60@gsm8k@llama-2-7b_quant@ID25670" 
ckpt_fnames = [fname for fname in os.listdir(model_dir) if fname.endswith(".bin") > 0]

for fname in ckpt_fnames:
    print(f"Processing {fname}")
    ckpt_file_path = os.path.join(model_dir, fname)
    fix_checkpoint(ckpt_file_path)
