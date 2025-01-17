import errno
import os
import io
from pathlib import Path
import select
from shutil import rmtree
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO
import argparse

extensions = ["mp3", "wav", "ogg", "flac"]  # we will look for all those file types.
# two_stems = None   # only separate one stems from the rest, for instance
two_stems = "vocals"

def find_files(in_path):
    out = []
    for file in Path(in_path).iterdir():
        if file.suffix.lower().lstrip(".") in extensions:
            out.append(file)
    return out

def copy_process_streams(process: sp.Popen):
    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())


    while fds:
        # `select` syscall will wait until one of the file descriptors has content.
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2 ** 16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()

def get_avg_volume(inp, outp, model):
    # Customize the following options!

    cmd = ["python3", "-m", "demucs.get_avg_volume", "-o", str(outp), "-n", model]

    if two_stems is not None:
        cmd += [f"--two-stems={two_stems}"]
    files = [str(f) for f in find_files(inp)]
    if not files:
        print(f"No valid audio files in {inp}")
        return
    p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()
    if p.returncode != 0:
        print("Command failed, something went wrong.")

def main(in_path,out_path,model):
    
    if not os.path.isdir(in_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), in_path)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
        print(f"{out_path} is created !!!")
    
    get_avg_volume(in_path,out_path,model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take audio files in input folder and Output the vocal volume results json file in output folder !")
    parser.add_argument("in_path", help="input folder path")
    parser.add_argument("out_path", help="output folder path")
    parser.add_argument("-n",
                        "--model",
                        type=str,
                        default="htdemucs",
                        help="pre-trained model type")
    args = parser.parse_args()
    main(args.in_path, args.out_path, args.model)