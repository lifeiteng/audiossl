
from audiossl.methods.atstframe.downstream.comparison_models.models.frame_atst import FrameAST_base
from torch import nn
import torch
from audiossl.methods.atstframe.downstream.utils_as_strong.model_as_strong import LinearHead
import torch
import torchaudio
from audiossl.transforms.common import MinMax
from torchvision import transforms
import sys



Labels = ['/m/032s66', '/m/01h3n', '/m/02g901', '/t/dd00013', '/m/07rrh0c', '/m/07p9k1k', '/m/0dgbq', '/m/09l8g', '/m/014zdl', '/m/07rrlb6', '/m/07qjznt', '/m/018p4k', '/m/0ghcn6', '/m/0284vy3', '/m/01m2v', '/m/07qf0zm', '/m/068zj', '/m/07q2z82', '/m/02p3nc', '/m/07prgkl', '/m/07pp_mv', '/m/07rgt08', '/m/01280g', '/m/0c2wf', '/m/02l6bg', '/m/0dxrf', '/m/0d31p', '/m/02mk9', '/m/02rtxlg', '/m/07rcgpl', '/m/07rpkh9', '/m/018w8', '/m/05x_td', '/m/093_4n', '/m/0bt9lr', '/m/053hz1', '/m/081rb', '/m/05zppz', '/m/01xq0k1', '/t/dd00001', '/m/07phxs1', '/m/0790c', '/m/09ct_', '/m/0641k', '/m/07qn4z3', '/m/0642b4', '/m/07m2kt', '/m/01c194', '/m/01j3sz', '/m/06bz3', '/m/0l156k', '/m/07rjzl8', '/m/07ptzwd', '/m/07qlwh6', '/m/01jt3m', '/m/07mzm6', '/m/05rj2', '/m/01w250', '/t/dd00067', '/m/01s0vc', '/m/032n05', '/m/0fx9l', '/m/03q5_w', '/m/07c52', '/m/07qrkrw', '/m/01swy6', '/m/07r660_', '/m/01j423', '/m/07pb8fc', '/m/02pjr4', '/m/01rd7k', '/t/dd00091', '/m/078jl', '/m/0c1dj', '/m/07pbtc8', '/m/07qcx4z', '/m/01x3z', '/m/03j1ly', '/m/012n7d', '/m/0ltv', '/m/07r5c2p', '/m/05tny_', '/m/0llzx', '/t/dd00088', '/m/07st88b', '/m/07sq110', '/m/07r4k75', '/m/01yg9g', '/m/06bxc', '/m/07pjjrj', '/m/07phhsh', '/m/0lyf6', '/m/01h82_', '/m/0h0rv', '/m/0195fx', '/t/dd00139', '/m/07sx8x_', '/m/07q6cd_', '/m/03k3r', '/m/025_jnm', '/m/02_nn', '/m/07r04', '/m/03kmc9', '/m/02dgv', '/m/0c1tlg', '/m/03w41f', '/m/01z5f', '/m/07qfgpx', '/m/0ch8v', '/m/02c8p', '/m/07rwm0c', '/m/07plz5l', '/m/027m70_', '/m/08j51y', '/m/01lynh', '/m/0fqfqc', '/m/07szfh9', '/m/046dlr', '/m/09x0r', '/m/0b_fwt', '/m/0l14jd', '/m/030rvx', '/m/01g50p', '/m/07s04w4', '/m/0btp2', '/m/02yds9', '/m/07q5rw0', '/m/07rv9rh', '/m/07rn7sz', '/t/dd00066', '/m/04_sv', '/m/0k65p', '/m/07rqsjt', '/m/0jbk', '/m/07pjwq1', '/m/09f96', '/t/dd00126', '/m/06hck5', '/m/02zsn', '/m/0ytgt', '/m/07r67yg', '/m/04fgwm', '/m/015lz1', '/m/07s2xch', '/m/04qvtq', '/m/01b9nn', '/m/06q74', '/m/0h9mv', '/t/dd00006', '/m/07qnq_y', '/m/07r10fb', '/t/dd00002', '/m/01bjv', '/m/015p6', '/m/07q0h5t', '/m/0dl83', '/m/06hps', '/m/01m4t', '/m/03cczk', '/m/0dv5r', '/m/01b_21', '/t/dd00077', '/m/06mb1', '/m/024dl', '/m/04cvmfc', '/m/0jb2l', '/m/0463cq4', '/m/056ks2', '/m/04rlf', '/m/07q0yl5', '/t/dd00127', '/m/0dv3j', '/m/07qwyj0', '/t/dd00112', '/m/0cdnk', '/m/07qv_x_', '/m/07qs1cx', '/m/068hy', '/m/07bjf', '/m/07qc9xj', '/m/0dl9sf8', '/m/03vt0', '/m/07jdr', '/m/07pl1bw', '/m/0g12c5', '/t/dd00125', '/m/0d4wf', '/m/07qb_dv', '/m/07rdhzs', '/m/07qcpgn', '/m/04zmvq', '/t/dd00005', '/m/07q8f3b', '/m/01g90h', '/m/07pdjhy', '/m/034srq', '/m/07pn_8q', '/m/012ndj', '/m/03qtwd', '/t/dd00048', '/m/07qwf61', '/m/0f8s22', '/t/dd00004', '/m/03qc9zr', '/m/0l7xg', '/m/020bb7', '/m/02y_763', '/m/03v3yw', '/t/dd00130', '/m/01hnzm', '/m/07pqn27', '/m/04s8yn', '/m/096m7z', '/m/03dnzn', '/t/dd00141', '/m/012xff', '/m/07st89h', '/m/028ght', '/m/0l15bq', '/m/01d3sd', '/m/073cg4', '/m/07pczhz', '/m/07ppn3j', '/m/07sk0jz', '/m/01y3hg', '/m/0gvgw0', '/m/02_41', '/m/09d5_', '/m/04fq5q', '/m/02mfyn', '/t/dd00135', '/m/07r_80w', '/m/0bzvm2', '/m/07qlf79', '/m/0gy1t2s', '/m/06xkwv', '/m/0chx_', '/m/01jnbd', '/m/07qqyl4', '/m/07r81j2', '/m/0j6m2', '/m/0199g', '/m/039jq', '/m/06h7j', '/m/07s0dtb', '/m/012f08', '/m/07p6mqd', '/m/07n_g', '/t/dd00092', '/m/06d_3', '/m/07qn5dc', '/m/0h2mp', '/m/03l9g', '/m/0239kh', '/m/06wzb', '/m/01jg02', '/m/02z32qm', '/m/0316dw', '/m/07r_25d', '/m/07pyy8b', '/t/dd00128', '/m/05zc1', '/t/dd00038', '/m/03p19w', '/m/0j2kx', '/m/0k5j', '/m/0hsrw', '/m/07pp8cl', '/m/0bcdqg', '/m/02bm9n', '/m/01d380', '/m/07p6fty', '/m/07k1x', '/m/01sb50', '/m/015jpf', '/m/0395lw', '/m/07qyrcz', '/t/dd00109', '/m/03fwl', '/m/07pqc89', '/m/07rbp7_', '/m/01z47d', '/m/04zjc', '/m/07rc7d9', '/m/025wky1', '/t/dd00065', '/m/0brhx', '/m/07s02z0', '/m/09b5t', '/m/07bgp', '/m/07pzfmf', '/m/02rr_', '/t/dd00003', '/m/01lsmm', '/m/07p7b8y', '/t/dd00136', '/m/04229', '/t/dd00142', '/m/0150b9', '/m/04rmv', '/m/01h8n0', '/m/04brg2', '/m/023vsd', '/m/0_1c', '/m/01j4z9', '/m/09ddx', '/m/07pxg6y', '/m/05mxj0q', '/m/0fw86', '/m/07qv4k0', '/m/0939n_', '/m/02fs_r', '/m/03wwcy', '/m/07yv9', '/m/03cl9h', '/m/07pc8lb', '/m/07s8j8t', '/m/06_y0by', '/m/07qh7jl', '/m/0174k2', '/m/02bk07', '/m/07qsvvw', '/m/07pt_g0', '/m/07rjwbb', '/m/03m9d0z', '/m/0dgw9r', '/m/07pws3f', '/m/07plct2', '/m/07r4gkf', '/m/0838f', '/m/07s34ls', '/m/07swgks', '/m/02jz0l', '/m/07r_k2n', '/m/07s12q4', '/m/07q4ntr', '/m/0g6b5', '/m/05kq4', '/m/0md09', '/m/02rhddq', '/m/02fxyj', '/m/07ptfmf', '/m/09ld4', '/m/0cmf2', '/m/0242l', '/m/07pdhp0', '/m/02qldy', '/m/05_wcq', '/m/09xqv', '/m/01hsr_', '/m/083vt', '/t/dd00121', '/m/0912c9', '/m/01v_m0', '/m/01yrx', '/m/07qjznl', '/m/07r4wb8', '/m/0261r1', '/m/02rlv9', '/m/0_ksk', '/m/07qmpdm', '/m/07sr1lc', '/m/07pyf11', '/m/07qfr4h', '/m/07qw_06', '/t/dd00134', '/m/07qdb04', '/m/04k94', '/m/02x984l', '/m/0k4j', '/m/0c3f7m', '/t/dd00061', '/m/03wvsk', '/m/0zmy2j9', '/m/07qz6j3', '/t/dd00018', '/m/014yck', '/m/07rwj3x', '/m/02p01q', '/m/09hlz4', '/t/dd00138', '/m/026fgl', '/m/0ngt1', '/m/025rv6n', '/m/02021', '/m/07rv4dm', '/m/07qwdck', '/m/019jd', '/m/0hgq8df', '/m/07q7njn', '/m/0hdsk', '/m/04gy_2', '/t/dd00143', '/m/02f9f_', '/g/122z_qxw', '/m/07cx4', '/m/04gxbd', '/m/07ryjzk', '/t/dd00099', '/m/01b82r', '/m/07pggtn', '/m/0130jx', '/m/07svc2k', '/m/07r5v4s', '/m/07rgkc5', '/m/06_fw', '/m/023pjk']





class InferenceAudioSetStrong(nn.Module):
    def __init__(self,ckpt_path):
        super().__init__()
        self.encoder = FrameAST_base()
        self.head = LinearHead(768, 407, use_norm=False, affine=False)
        self._load_ckpt(ckpt_path)
        self.transform = self._transform()

    def _transform(self):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            16000, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        normalize = MinMax(min=-79.6482,max=50.6842)
        return transforms.Compose([melspec_t,
                                to_db,
                                normalize])

    
    def _load_ckpt(self,ckpt_path):
        s = torch.load(ckpt_path,map_location="cpu")
        state_dict = s["state_dict"]
        replaced_state_dict = {}
        for key in state_dict.keys():
            replaced_state_dict[key.replace("encoder.encoder","encoder")] =  state_dict[key]

        self.load_state_dict(replaced_state_dict)

    def predict(self,wav):
        """
        ==================================================
        args:
        wav: torch.tensor in the shape of [1,N] or [B,1,N] 
        """"""
        return:
             retured prediction in the shape of [1,407,T] or [B,407,T]
        """
        if len(wav.shape)==2: 
            wav = wav.unsqueeze(1)
        else:
            assert len(wav.shape) == 3
    

        mel = self.transform(wav)
        chunk_len=1001 #10 secnods, consistent with the length of positional embedding
        output = []

        total_len = mel.shape[-1]
        num_chunks = total_len // chunk_len + 1
        for i in range(num_chunks):

            start = i*chunk_len
            end = (i+1) * chunk_len
            if end > total_len:
                end = total_len
            if end>start:
                mel_chunk=mel[:,:,:,start:end]
                len_chunk = torch.tensor([mel_chunk.shape[-1]]).expand(mel.shape[0]).to(wav.device)

                output_chunk = self.encoder.get_intermediate_layers(mel_chunk,len_chunk,n=1,scene=False)

                output.append(output_chunk)
        output=torch.cat(output,dim=1)
        output = self.head(output)
        return output

import os
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


import csv
from collections import defaultdict
import pandas as pd
import logging
formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
logging.basicConfig(format=formatter, level=logging.INFO)

def parse_audioset_strong_merged(merged_file, labels, labelmap):
    merged2mids = defaultdict(lambda: [])

    pf = pd.read_csv(merged_file)
    # index,mid,display_name,remarks,annotation_count,node_level,demension1,dimension2,dimension3,dimension4,dimension5,dimension6
    used = set()
    for x in range(1, 7):
        if f"dimension{x}" not in pf.columns:
            break
        names = set(pf[f"dimension{x}"].dropna().values.tolist())
        for name in names:
            mids = pf[pf[f"dimension{x}"] == name]["mid"].values.tolist()
            assert name not in merged2mids
            merged2mids[name].extend(mids)
            used.update(set(mids))

    not_used = set(labels) - used
    logging.info(f"Number of merged labels: {len(merged2mids)} not used: {len(not_used)}")
    transform = np.zeros((len(labels), len(merged2mids) + len(not_used)), dtype=np.float32)
    keys = sorted(list(merged2mids.keys()))
    
    num_active = 0
    for k, name in enumerate(keys):
        mids = merged2mids[name]
        for mid in mids:
            try:
                idx = labels.index(mid)
                transform[idx, k] = 1.0
                num_active += 1
            except:
                pass
    not_used = sorted(list(not_used))
    for k, mid in enumerate(not_used):
        idx = labels.index(mid)
        transform[idx, len(merged2mids) + k] = 1.0
        num_active += 1
    keys += [f">>{labelmap[mid]}" for mid in not_used]
    assert num_active > 100
    return transform, keys


import argparse
def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    import sys
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from audiossl.datasets.as_strong_utils.as_strong_dict import get_lab_dict
    from pyannote.audio.utils.signal import binarize_ndarray

    parser = argparse.ArgumentParser(description="Eval SED")
    # parser.add_argument("--merge", type=str2bool, default=False, help="Merge labels")
    parser.add_argument("--merged_file", type=str, default="", help="Merged labels file")
    parser.add_argument("--model_file", type=str, default="atst_ft_StrongAS_eps28.ckpt", help="model checkpoint file")
    parser.add_argument("--binarize", type=str2bool, default=False, help="Plot binarized flag")
    parser.add_argument("--suffix", type=str, default="", help="output file suffix")
    parser.add_argument("--audio_paths", nargs='+', help="audio paths")
    
    args = parser.parse_args()

    labelmap, labels = {}, []
    with open("./mid_to_display_name.tsv") as f:
        for line in f:
            key, name = line.strip().split('\t')
            assert key not in labelmap
            labelmap[key] = name

    # Labels = [line.strip() for line in open("./common_labels.txt").readlines()]
    for l in Labels:
        labels.append(labelmap[l])

    # classes_labels = get_lab_dict("./common_labels.txt")
    # for l in list(classes_labels.keys()):
    #     labels.append(labelmap[l])

    print(f"Labels {len(labels)}")

    if args.merged_file:
        # merged_file = "/Users/feiteng/StarQuestAI/QuestarLLM/FAlignerBenchmark/audioset/strong_456_class_labels_indices_merged.csv"
        transform, merged_labels = parse_audioset_strong_merged(args.merged_file, Labels, labelmap)

    ckpt_path, audio_paths  = args.model_file, args.audio_paths

    m = InferenceAudioSetStrong(ckpt_path)
    m.eval()

    for audio_path in audio_paths:
        base_file = Path(audio_path).parent / Path(audio_path).stem
        
        (waveform, _) = librosa.core.load(audio_path, sr=16000, mono=True)
        waveform = waveform.reshape([1, -1])
        # print(waveform.shape)

        with torch.no_grad():
            framewise_output = m.predict(torch.from_numpy(waveform))
        framewise_output = framewise_output[0].detach().cpu().numpy().transpose(1, 0)
        logging.info(f"FRMAE {framewise_output.shape}")

        sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

        frames_per_second = 16000 // 160


        top_k = 10  # Show top results
        sorted_indexes_max = np.argsort(np.max(framewise_output, axis=0))[::-1][0 : top_k]
        sorted_indexes_sum = np.argsort(np.sum(framewise_output, axis=0))[::-1][0 : top_k]

        # sorted_indexes = np.concatenate([sorted_indexes_max, sorted_indexes_sum])
        # sorted_indexes = list(set(sorted_indexes.tolist()))

        sorted_indexes = sorted_indexes_max.tolist()
        for idx in sorted_indexes_sum.tolist():
            if idx not in sorted_indexes:
                sorted_indexes.append(idx)


        """(time_steps, top_k)"""
        # prob_path = f'{base_file}{args.suffix}_probs.txt'
        # np.savetxt(prob_path, top_result_mat.T, fmt='%.3e')

        logging.info(f"Sorted Indexes: {sorted_indexes[0 : top_k]}")
        logging.info(f"Sorted  Labels: {[Labels[k] for k in sorted_indexes[0 : top_k]]}")
        logging.info(f"Sorted  labels: {[labels[k] for k in sorted_indexes[0 : top_k]]}")


        # Plot result
        # Figure 1
        stft = librosa.core.stft(y=waveform[0], n_fft=1024, hop_length=160, window='hann', center=True)
        frames_num = stft.shape[-1]
        logging.info(f"STFT {stft.shape}")

        # 扩展开 frames
        framewise_output = framewise_output.repeat(frames_num // framewise_output.shape[0], axis=0)

        top_result_mat = framewise_output[:, sorted_indexes]

        if args.merged_file:
            fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 1, 1.1]})
        elif args.binarize:
            fig, axs = plt.subplots(3, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 1, 1]})
        else:
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

        vv = axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
        axs[0].set_ylabel('Frequency bins')
        axs[0].set_title('Log spectrogram')
        plt.colorbar(vv, ax=axs[0])

        # Figure 2
        logging.info(f"   raw {top_result_mat.shape} v[{top_result_mat.min():.2f}, {top_result_mat.max():.2f}]")
        vv = axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
        # plot line at top_k
        axs[1].axhline(y=top_k-0.5, xmin=0.0, xmax=1.0, color='r')
        axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
        # axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
        axs[1].yaxis.set_ticks(np.arange(0, len(sorted_indexes)))
        axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes])
        # axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
        # axs[1].set_xlabel('Seconds')
        # axs[1].xaxis.set_ticks_position('bottom')
        plt.colorbar(vv, ax=axs[1])

        if args.binarize:
            # 移动平均
            from torch.nn import functional as F
            kernel_size = 11
            probs = F.avg_pool1d(torch.tensor(top_result_mat.T).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze(0).numpy()

            binarized = binarize_ndarray(probs, onset=0.1, offset=0.1)
            vv = axs[2].matshow(binarized.astype(np.float32) * 0.5, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
            axs[2].axhline(y=top_k-0.5, xmin=0.0, xmax=1.0, color='r')
            axs[2].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
            axs[2].yaxis.set_ticks(np.arange(0, len(sorted_indexes)))
            axs[2].yaxis.set_ticklabels(np.array(labels)[sorted_indexes])
            plt.colorbar(vv, ax=axs[2])

        if args.merged_file:
            # Figure 3
            framewise_output = np.dot(framewise_output, transform)

            sorted_indexes_max = np.argsort(np.max(framewise_output, axis=0))[::-1][0 : top_k]
            sorted_indexes_sum = np.argsort(np.sum(framewise_output, axis=0))[::-1][0 : top_k]
            sorted_indexes = sorted_indexes_max.tolist()
            for idx in sorted_indexes_sum.tolist():
                if idx not in sorted_indexes:
                    sorted_indexes.append(idx)
            top_result_mat = framewise_output[:, sorted_indexes]

            logging.info(f"merged {top_result_mat.shape} v[{top_result_mat.min():.2f}, {top_result_mat.max():.2f}]")

            vv = axs[2].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
            # plot line at top_k
            axs[2].axhline(y=top_k-0.5, xmin=0.0, xmax=1.0, color='r')
            axs[2].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
            # axs[2].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
            axs[2].yaxis.set_ticks(np.arange(0, len(sorted_indexes)))
            axs[2].yaxis.set_ticklabels(np.array(merged_labels)[sorted_indexes])
            # axs[2].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
            # axs[2].set_xlabel('Seconds')
            # axs[2].xaxis.set_ticks_position('bottom')
            plt.colorbar(vv, ax=axs[2])

        plt.tight_layout()
        fig_path = f'{base_file}{args.suffix}.png'
        plt.savefig(fig_path)
        logging.info('Save sound event detection visualization to {}'.format(fig_path))
