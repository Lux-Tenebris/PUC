#include "Block.h"
#include "sha256.h"
#include <iostream>
#include <omp.h>

Block::Block(uint32_t nIndexIn, const string &sDataIn) : _nIndex(nIndexIn), _sData(sDataIn)
{
    _nNonce = 0;
    _tTime = time(nullptr);

    sHash = _CalculateHash();
}


void Block::MineBlock(uint32_t nDifficulty)
{
    char cstr[nDifficulty + 1];
    std::cout << "Loop 1 - MineBlock - Block.h" << std::endl;
    for (uint32_t i = 0; i < nDifficulty; ++i)
    {
        cstr[i] = '0';
    }
    cstr[nDifficulty] = '\0';

    string str(cstr);
    string targetHash(nDifficulty, '0');

    uint32_t threadNonce;
    string threadHash;

    uint32_t numThreads = 10;  // Definir o número de threads

    #pragma omp target teams num_teams(numThreads) thread_limit(1) map(tofrom: threadHash, threadNonce)
    {
        uint32_t threadId = omp_get_thread_num();

        #pragma omp parallel num_threads(1) reduction(min : threadHash, threadNonce)
        {
            threadNonce = _nNonce + threadId;

            #pragma omp for
            for (uint32_t nonce = threadNonce; nonce < UINT32_MAX; nonce += numThreads)
            {
                threadHash = CalculateHashWithNonce(nonce);
                if (threadHash.substr(0, nDifficulty) == targetHash)
                {
                    #pragma omp critical
                    {
                        if (threadHash < sHash)
                        {
                            sHash = threadHash;
                            _nNonce = nonce;
                        }
                    }
                    break;
                }
            }
        }
    }

    std::cout << "Block mined: " << sHash << std::endl;
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
