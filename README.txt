This repo implements a bin-packing optimizer as described in this paper: https://www.cs.rit.edu/~ark/students/amb4757/report.pdf written in C++11 and CUDA.
This project was Mine and Maxwell Johnson's final project for 15-418 at CMU (Parallel Architecture).

After cloning the repo, run:
git clone https://github.com/open-source-parsers/jsoncpp.git
cd jsoncpp
python2 ./amalgamate.py
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/xyz"
make

Where xyz is the pwd from root (you'll need to append this to your shell's init script if you want this environment variable to exist beyond your shell's lifetime)
