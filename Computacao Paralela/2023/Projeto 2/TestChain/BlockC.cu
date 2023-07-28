#include "Block.h"
#include "sha256.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

Block::Block(uint32_t nIndexIn, const string &sDataIn) : _nIndex(nIndexIn), _sData(sDataIn)
{   
    _nNonce = 0;
    _tTime = time(nullptr);
    
    sHash = _CalculateHash();
}

__global__ void MineBlockKernel(uint32_t nDifficulty, char* cstr, uint32_t* nonce, uint32_t* bestNonce, char* sHash)
{
    uint32_t threadId = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t numThreads = blockDim.x * gridDim.x;
    uint32_t threadNonce = *nonce + threadId;
    char targetHash[nDifficulty + 1];

    for (uint32_t i = 0; i < nDifficulty; ++i)
    {
        targetHash[i] = '0';
    }
    targetHash[nDifficulty] = '\0';

    do {
        threadNonce += numThreads;
        string threadHash = CalculateHashWithNonce(threadNonce);

        if (threadHash.substr(0, nDifficulty) == targetHash) {
            atomicMin(bestNonce, threadNonce);
        }

    } while (*bestNonce > threadNonce);

    if (*bestNonce == threadNonce) {
        *nonce = threadNonce;
        *sHash = CalculateHashWithNonce(*nonce);
    }
}

void Block::MineBlock(uint32_t nDifficulty)
{
    char cstr[nDifficulty + 1];
    for (uint32_t i = 0; i < nDifficulty; ++i)
    {
        cstr[i] = '0';
    }
    cstr[nDifficulty] = '\0';

    string targetHash(nDifficulty, '0');

    uint32_t* dev_nonce;
    char* dev_sHash;
    uint32_t* dev_bestNonce;
    cudaMalloc((void**)&dev_nonce, sizeof(uint32_t));
    cudaMalloc((void**)&dev_sHash, sizeof(char) * 65);
    cudaMalloc((void**)&dev_bestNonce, sizeof(uint32_t));
    cudaMemcpy(dev_bestNonce, &_nNonce, sizeof(uint32_t), cudaMemcpyHostToDevice);

    while (true) {
        cudaMemcpy(dev_nonce, &_nNonce, sizeof(uint32_t), cudaMemcpyHostToDevice);
        MineBlockKernel<<<numBlocks, threadsPerBlock>>>(nDifficulty, cstr, dev_nonce, dev_bestNonce, dev_sHash);
        cudaMemcpy(&_nNonce, dev_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(sHash, dev_sHash, sizeof(char) * 65, cudaMemcpyDeviceToHost);
        if (sHash.substr(0, nDifficulty) == targetHash) {
            break;
        }
    }

    cudaFree(dev_nonce);
    cudaFree(dev_sHash);
    cudaFree(dev_bestNonce);

    cout << "Block mined: " << sHash << endl;
}

string Block::CalculateHashWithNonce(uint32_t nonce) const
{
        stringstream ss;
        ss << _nIndex << sPrevHash << _tTime << _sData << nonce;
        return sha256(ss.str());
}

inline string Block::_CalculateHash() const
{
    stringstream ss;
    ss << _nIndex << sPrevHash << _tTime << _sData << _nNonce;

    return sha256(ss.str());
}
