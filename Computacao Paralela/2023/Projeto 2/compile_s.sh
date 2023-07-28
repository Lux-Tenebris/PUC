#!/bin/bash

gcc8 -lstdc++ -o testchain -std=c++11 -x c++ TestChain/main.cpp TestChain/Block.cpp TestChain/Blockchain.cpp TestChain/sha256.cpp
