import os
import pandas as pd
import soundfile as sf
from argparse import ArgumentParser

parser = ArgumentParser("Data Preprocess")
parser.add_argument("--audio_path", type=str, required=True)
parser.add_argument("--meta_path", type=str, required=True)
args = parser.parse_args()

meta_path = args.meta_path
os.chdir(meta_path) 
print(f"meta_path = {meta_path}")

if not os.path.exists("./train"):
    os.mkdir("./train")
if not os.path.exists("./eval"):
    os.mkdir("./eval")

# --zip_audios
# ├── balanced_train_segments
# ├── eval_segments
# └── unbalanced_train_segments

audio_files = {}
for sub in ["eval_segments", "balanced_train_segments", "unbalanced_train_segments"]:
    file_list = os.listdir(f"{args.audio_path}/{sub}/")
    for file in file_list:
        if file.endswith(".wav"):
            assert file not in audio_files, file
            audio_files[file] = f"{sub}/{file}"

mode = ["eval", "train"]
# standard dcase label format: filename	onset	offset	event_label


# rxhnnJ_hJyk_30000 -> rxhnnJ_hJyk
def get_youtube_id(filename):
    return "Y" + "_".join(filename.split("_")[:-1]) + ".wav"

for m in mode:
    num_print = 0
    full_meta_df = pd.read_csv(f"./source/audioset_{m}_strong.csv", delimiter="\t")

    if m == "eval":
        full_file_list = full_meta_df['segment_id'].values
        full_file_mask = []
        mask2exist = {}
        for file in full_file_list:
            if file not in mask2exist:
                mask2exist[file] = get_youtube_id(file) in audio_files
            if not mask2exist[file] and num_print < 10:
                num_print += 1
                print(f"File {file} not exist")

            full_file_mask.append(mask2exist[file])
        print(f"Exist {sum([v for k, v in mask2exist.items()])}/{len(mask2exist)} wav files")

        meta_df = full_meta_df[full_file_mask]
        meta_df.columns = ["filename", "onset", "offset", "event_label"]
        meta_df["filename"] = meta_df["filename"].map(lambda x: audio_files[get_youtube_id(x)])
    else:
        meta_df = full_meta_df
    
        meta_df.columns = ["filename", "onset", "offset", "event_label"]
        meta_df["filename"] = meta_df["filename"].map(lambda x: get_youtube_id(x))
    

    meta_df.to_csv(f"./{m}/{m}.tsv", index=False, sep="\t")
    print(f"Total {m} files after filtering:", len(meta_df))

# generate duration tsv for eval data
eval_meta = pd.read_csv(f"./eval/eval.tsv", delimiter="\t")
file_list = pd.unique(eval_meta["filename"].values)
durations = []
for file in file_list:
    wav, fs = sf.read(f"{args.audio_path}/{file}")
    durations.append(min(10, len(wav) / fs))
duration_df = pd.DataFrame({"filename": file_list, "duration": durations})
duration_df.to_csv("./eval/eval_durations.tsv", index=False, sep="\t")

