# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from pathlib import Path
import subprocess

from dora.log import fatal
import torch as th
import torchaudio as ta
import numpy as np
import json
from math import log10
from tqdm import tqdm


from .apply import apply_model, BagOfModels
from .audio import AudioFile, convert_audio
from .htdemucs import HTDemucs
from .pretrained import get_model_from_args, add_model_flags, ModelLoadingError


def load_track(track, audio_channels, samplerate):
    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(
            streams=0,
            samplerate=samplerate,
            channels=audio_channels)
    except FileNotFoundError:
        errors['ffmpeg'] = 'FFmpeg is not installed.'
    except subprocess.CalledProcessError:
        errors['ffmpeg'] = 'FFmpeg could not read the file.'

    if wav is None:
        try:
            wav, sr = ta.load(str(track))
        except RuntimeError as err:
            errors['torchaudio'] = err.args[0]
        else:
            wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        print(f"Could not load file {track}. "
              "Maybe it is not a supported file format? ")
        for backend, error in errors.items():
            print(f"When trying to load using {backend}, got the following error: {error}")
        sys.exit(1)
    return wav


def get_parser():
    parser = argparse.ArgumentParser("demucs.separate",
                                     description="Separate the sources for the given tracks")
    parser.add_argument("tracks", nargs='+', type=Path, default=[], help='Path to tracks')
    add_model_flags(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o",
                        "--out",
                        type=Path,
                        default=Path("separated"),
                        help="Folder where to put extracted tracks. A subfolder "
                        "with the model name will be created.")
    parser.add_argument("--filename",
                        default="{track}/{stem}.{ext}",
                        help="Set the name of output file. \n"
                        'Use "{track}", "{trackext}", "{stem}", "{ext}" to use '
                        "variables of track name without extension, track extension, "
                        "stem name and default output file extension. \n"
                        'Default is "{track}/{stem}.{ext}".')
    parser.add_argument("-d",
                        "--device",
                        default="cuda" if th.cuda.is_available() else "cpu",
                        help="Device to use, default is cuda if available else cpu")
    parser.add_argument("--shifts",
                        default=1,
                        type=int,
                        help="Number of random shifts for equivariant stabilization."
                        "Increase separation time but improves quality for Demucs. 10 was used "
                        "in the original paper.")
    parser.add_argument("--overlap",
                        default=0.25,
                        type=float,
                        help="Overlap between the splits.")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--no-split",
                             action="store_false",
                             dest="split",
                             default=True,
                             help="Doesn't split audio in chunks. "
                             "This can use large amounts of memory.")
    split_group.add_argument("--segment", type=int,
                             help="Set split size of each chunk. "
                             "This can help save memory of graphic card. ")
    parser.add_argument("--two-stems",
                        dest="stem", metavar="STEM",
                        help="Only separate audio into {STEM} and no_{STEM}. ")
    parser.add_argument("-j", "--jobs",
                    default=0,
                    type=int,
                    help="Number of jobs. This can increase memory usage but will "
                            "be much faster when multiple cores are available.")


    return parser


def main(opts=None):
    parser = get_parser()
    args = parser.parse_args(opts)

    try:
        model = get_model_from_args(args)
    except ModelLoadingError as error:
        fatal(error.args[0])

    max_allowed_segment = float('inf')
    if isinstance(model, HTDemucs):
        max_allowed_segment = float(model.segment)
    elif isinstance(model, BagOfModels):
        max_allowed_segment = model.max_allowed_segment
    if args.segment is not None and args.segment > max_allowed_segment:
        fatal("Cannot use a Transformer model with a longer segment "
              f"than it was trained for. Maximum segment is: {max_allowed_segment}")

    if isinstance(model, BagOfModels):
        print(f"Selected model is a bag of {len(model.models)} models. "
              "You will see that many progress bars per track.")

    model.cpu()
    model.eval()

    if args.stem is not None and args.stem not in model.sources:
        fatal(
            'error: stem "{stem}" is not in selected model. STEM must be one of {sources}.'.format(
                stem=args.stem, sources=', '.join(model.sources)))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"Separated tracks will be stored in {out.resolve()}")
    
    results = {}

    for track in tqdm(args.tracks):
        if not track.exists():
            print(
                f"File {track} does not exist. If the path contains spaces, "
                "please try again after surrounding the entire path with quotes \"\".",
                file=sys.stderr)
            continue
        wav = load_track(track, model.audio_channels, model.samplerate)

        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        sources = apply_model(model, wav[None], device=args.device, shifts=args.shifts,
                              split=args.split, overlap=args.overlap, progress=False,
                              num_workers=args.jobs, segment=args.segment)[0]

        sources *= ref.std()
        sources += ref.mean()

        sources_list = list(sources)
        waveforms = sources_list.pop(model.sources.index(args.stem))
        waveforms = waveforms.data.cpu().numpy()

        volumes = [np.sum(waveform ** 2)/len(waveform) for waveform in waveforms]
        average_volume = np.mean(volumes)
        avg_db = 10*log10(average_volume)
        results[str(track.name.rsplit(".", 1)[0])]=avg_db


    with open(str(out)+"/results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
