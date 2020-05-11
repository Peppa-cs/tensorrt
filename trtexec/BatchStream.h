/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include "NvInfer.h"
#include "common.h"
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <vector>

using namespace std;
class IBatchStream
{
public:
    virtual void reset(int firstBatch) = 0;
    virtual bool next() = 0;
    virtual void skip(int skipCount) = 0;
    virtual float* getBatch() = 0;
    virtual float* getLabels() = 0;
    virtual int getBatchesRead() const = 0;
    virtual int getBatchSize() const = 0;
    virtual nvinfer1::Dims getDims() const = 0;
};

struct SampleINT8APIPreprocessing
{
    // Preprocessing values are available here: https://github.com/onnx/models/tree/master/models/image_classification/resnet
    std::vector<float> mean {0.485f, 0.456f, 0.406f};
    std::vector<float> std {0.229f, 0.224f, 0.225f};
    float scale{255.0f};
    std::vector<int> inputDims{3,224,224};
};
struct vggPreprocessing
{
 //   std::vector<float> mean {104.0f, 117.0f, 123.0f};
std::vector<float> mean {123.0f, 117.0f, 104.0f};
    float scale{0.0078125f};

};

struct cifar10
{
    std::vector<float> mean {0.4914f, 0.4822f, 0.4465f};
    std::vector<float> std {0.2023f, 0.1994f, 0.2010f};
    float scale{255.0f};
    std::vector<int> inputDims{3,32,32};
};

class MNISTBatchStream : public IBatchStream
{
public:
    MNISTBatchStream(int batchSize, int maxBatches, const std::string& dataFile, const std::string& labelsFile,
        const std::vector<std::string>& directories)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{3, 1, 28, 28} //!< We already know the dimensions of MNIST images.
    {
        readDataFile(locateFile(dataFile, directories));
        readLabelsFile(locateFile(labelsFile, directories));
    }

    void reset(int firstBatch) override
    {
        mBatchCount = firstBatch;
    }

    bool next() override
    {
        if (mBatchCount >= mMaxBatches)
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
        mBatchCount += skipCount;
    }

    float* getBatch() override
    {
     
        return mData.data() + (mBatchCount * mBatchSize * samplesCommon::volume(mDims));
    }

    float* getLabels() override
    {
        return mLabels.data() + (mBatchCount * mBatchSize);
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override
    {
        return mDims;
    }
        nvinfer1::Dims getImageDims() 
    {
        return mDims;
    }

private:
    void readDataFile(const std::string& dataFilePath)
    {
        std::ifstream file{dataFilePath.c_str(), std::ios::binary};

        int magicNumber, numImages, imageH, imageW;
        file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        // All values in the MNIST files are big endian.
        magicNumber = samplesCommon::swapEndianness(magicNumber);
        assert(magicNumber == 2051 && "Magic Number does not match the expected value for an MNIST image set");

        // Read number of images and dimensions
        file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        file.read(reinterpret_cast<char*>(&imageH), sizeof(imageH));
        file.read(reinterpret_cast<char*>(&imageW), sizeof(imageW));

        numImages = samplesCommon::swapEndianness(numImages);
        imageH = samplesCommon::swapEndianness(imageH);
        imageW = samplesCommon::swapEndianness(imageW);

        // The MNIST data is made up of unsigned bytes, so we need to cast to float and normalize.
        int numElements = numImages * imageH * imageW;
        std::vector<uint8_t> rawData(numElements);
        file.read(reinterpret_cast<char*>(rawData.data()), numElements * sizeof(uint8_t));
        mData.resize(numElements);
        std::transform(
            rawData.begin(), rawData.end(), mData.begin(), [](uint8_t val) { return static_cast<float>(val) / 255.f; });
    }

    void readLabelsFile(const std::string& labelsFilePath)
    {
        std::ifstream file{labelsFilePath.c_str(), std::ios::binary};
        int magicNumber, numImages;
        file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        // All values in the MNIST files are big endian.
        magicNumber = samplesCommon::swapEndianness(magicNumber);
        assert(magicNumber == 2049 && "Magic Number does not match the expected value for an MNIST labels file");

        file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        numImages = samplesCommon::swapEndianness(numImages);

        std::vector<uint8_t> rawLabels(numImages);
        file.read(reinterpret_cast<char*>(rawLabels.data()), numImages * sizeof(uint8_t));
        mLabels.resize(numImages);
        std::transform(
            rawLabels.begin(), rawLabels.end(), mLabels.begin(), [](uint8_t val) { return static_cast<float>(val); });
    }

    int mBatchSize{0};
    int mBatchCount{0}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    Dims mDims{};
    std::vector<float> mData{};
    std::vector<float> mLabels{};
};

//********************************************************************************************************//

class PPMcifarINT8Stream : public IBatchStream
{
public:
    PPMcifarINT8Stream(int batchSize, int maxBatches,nvinfer1::Dims Dims)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{Dims}  
    {
        readDataFile();
    //    readLabelsFile();
    }

    void reset(int firstBatch) override
    {
        mBatchCount = firstBatch;
    }

    bool next() override
    {
        if (mBatchCount == mMaxBatches-1 )
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
        mBatchCount += skipCount;
    }

    float* getBatch() override
    {   
        count++;

   //     int num = mMaxBatches/mBatchSize;
   //     if( mBatchCount % num == 0) {    gLogInfo << "count is " <<mBatchCount<<std::endl; count=0;}
        return mData.data() + (count * mBatchSize * samplesCommon::volume(mDims));

    }

    float* getLabels() override
    {
        return mLabels.data() + (mBatchCount * mBatchSize);
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }
    nvinfer1::Dims getDims() const override
    {
        return mDims;
    }
    nvinfer1::Dims getImageDims() 
    {
        return mDims;
    }
    int count{-1};
private:
    void readDataFile()
    {
    int channels =3;
    int height =32;
    int width =32;
    int max ;
    
    SampleINT8APIPreprocessing p;
    vector<uint8_t> fileData((mBatchSize*mMaxBatches)*channels * height * width);
//    vector<uint8_t> fileData(mMaxBatches*channels * height * width);    
    std::string magic{""};
    uint8_t *p_data;
    p_data = fileData.data();
    mData.resize((mBatchSize*mMaxBatches)*channels * height * width);
//    mData.resize(mMaxBatches*channels * height * width);
   vector<string> category;

   category.push_back("airplane");
   category.push_back("automobile");
   category.push_back("bird");
   category.push_back("cat");
   category.push_back("deer");
   category.push_back("dog");
   category.push_back("frog");
   category.push_back("horse");
   category.push_back("ship");
   category.push_back("truck");
 cifar10 normalize;
for(int i=0,k=-1,j=0; i < mBatchSize*mMaxBatches; i++,j++)
{

   if(i % 50 == 0) k++;
    if(i < 9)
   {

    std::ifstream infile("/home/peppa3/cifar/"+category[k]+"000"+std::to_string(i+1)+".ppm", std::ifstream::binary);

    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);
   }
    else if(i < 99)
   {
    std::ifstream infile("/home/peppa3/cifar/"+category[k]+"00"+std::to_string(i+1)+".ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);
    }
    else if(i < 999)
   { 
    std::ifstream infile("/home/peppa3/cifar/"+category[k]+"0"+std::to_string(i+1)+".ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);
   }
   else
   {
    std::ifstream infile("/home/peppa3/cifar/"+category[k]+std::to_string(i+1)+".ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);

   }

    p_data += width * height * channels;



     for (int h = 0; h < height; ++h)
      {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = j * channels *height *width + ( h * width * channels + w * channels + 2 );
                int srcIdx = j * channels *height *width + ( h * width * channels + w * channels );
                int temp = fileData[dstIdx];
                fileData[dstIdx] = fileData[srcIdx];
                fileData[srcIdx] = temp;
            }
      }

    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = j * channels *height *width + ( c * height * width + h * width + w );
                int srcIdx = j * channels *height *width + ( h * width * channels + w * channels + c );

                mData[dstIdx] = (float) fileData[srcIdx];
            }
        }
    }

    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = j * channels *height *width + ( c * height * width + h * width + w );

                mData[dstIdx] = (float) (mData[dstIdx] / normalize.scale -normalize.mean.at(c)) / normalize.std.at(c);
            }
        }
    }
   }

      std::cout<<"mysize is "<<mData.size()<<"success"<<std::endl;
    }

    void readLabelsFile()
    {
    vector<string> groundtruth;
   
    if (!samplesCommon::readReferenceFile("/usr/src/tensorrt/data/alexnet/groundtruth.txt", groundtruth))
    {
        gLogError << "Unable to read reference file: "  << std::endl;
            }

     for(int i = 0;i < mBatchSize*mMaxBatches;i++) mLabels.push_back(atof(groundtruth[i].c_str()));

     std::cout<<"size "<<mLabels.size()<<"   "<<mLabels[0]<<std::endl;
    }

    int mBatchSize{0};
    int mBatchCount{-1}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    Dims mDims{};
    std::vector<float> mData{};
    std::vector<float> mLabels{};
};


//********************************************************************************************************//

class BINcifarBatchStream 
{
public:
    BINcifarBatchStream(int batchSize, int maxBatches,nvinfer1::Dims Dims)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{Dims}  
    {
        readDataFile();
    //    readLabelsFile();
    }

    void reset(int firstBatch) 
    {
        mBatchCount = firstBatch;
    }

    bool next() 
    {
        if (mBatchCount == mMaxBatches-1 )
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) 
    {
        mBatchCount += skipCount;
    }

    float* getBatch() 
    {   
        count++;

   //     int num = mMaxBatches/mBatchSize;
   //     if( mBatchCount % num == 0) {    gLogInfo << "count is " <<mBatchCount<<std::endl; count=0;}
        return mData.data() + (count * mBatchSize * samplesCommon::volume(mDims));

    }

    char* getLabel() 
    {
      
        return mLabels.data() + (count * mBatchSize);
    }

    int getBatchesRead() const 
    {
        return mBatchCount;
    }

    int getBatchSize() const 
    {
        return mBatchSize;
    }
    nvinfer1::Dims getDims() const 
    {
        return mDims;
    }
    nvinfer1::Dims getImageDims() 
    {
        return mDims;
    }
    int count{-1};
private:
    void readDataFile()
    {
    int channels =3;
    int height =32;
    int width =32;

    vector<uint8_t> fileData((mBatchSize*mMaxBatches)*channels * height * width);
   
    mData.resize((mBatchSize*mMaxBatches)*channels * height * width);
    mLabels.resize(mBatchSize*mMaxBatches);
    uint8_t *p_data;
    p_data = fileData.data();
    char *l_data;
    l_data = mLabels.data();

    cifar10 normalize;

    ifstream fin;
    fin.open("/home/peppa3/test_batch.bin",ifstream::binary);
    assert(fin && "Attempting to read from a file that is not open.");
  //  FILE *fpr = fopen("/home/peppa3/test_batch.bin","rb");
    
    for(int i = 0;i< mBatchSize*mMaxBatches; i++)
    {
      //  gLogError << "i =  " <<i << std::endl;
       //fread(l_data,sizeof(char),1,fpr);
       //fread(p_data,sizeof(char),channels * height * width,fpr);
       fin.read(reinterpret_cast<char*>(l_data),1);
     //   gLogError << "l_data =  " <<std::to_string(mLabels[i]) << std::endl;
       fin.read(reinterpret_cast<char*>(p_data),channels * height * width); 

       l_data+=1;
       p_data += width * height * channels;
   // gLogError << "success 1  "  << std::endl;
   /*    for (int c = 0; c < channels; ++c)
      {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = i * channels *height *width + ( c * height * width + h * width + w );
                int srcIdx = i * channels *height *width + ( h * width * channels + w * channels + c );

                mData[dstIdx] = (float) fileData[srcIdx];
            }
        }
      }*/
   // gLogError << "success 2  "  << std::endl;
      for (int c = 0; c < channels; ++c)
     {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = i * channels *height *width + ( c * height * width + h * width + w );

                mData[dstIdx] = (float) (fileData[dstIdx] / normalize.scale -normalize.mean.at(c)) / normalize.std.at(c);
            }
        }
     }
    }
fin.close();
  //  fclose(fpr);
    gLogError << "success res  "  << std::endl;
    }

    void readLabelsFile()
    {

    }

    int mBatchSize{0};
    int mBatchCount{-1}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    Dims mDims{};
    std::vector<float> mData{};
    std::vector<char> mLabels{};
};
//********************************************************************************************************//

class PPMcifarBatchStream : public IBatchStream
{
public:
    PPMcifarBatchStream(int batchSize, int maxBatches,nvinfer1::Dims Dims)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{Dims}  
    {
        readDataFile();
    //    readLabelsFile();
    }

    void reset(int firstBatch) override
    {
        mBatchCount = firstBatch;
    }

    bool next() override
    {
        if (mBatchCount == mMaxBatches-1 )
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
        mBatchCount += skipCount;
    }

    float* getBatch() override
    {   
        count++;

   //     int num = mMaxBatches/mBatchSize;
   //     if( mBatchCount % num == 0) {    gLogInfo << "count is " <<mBatchCount<<std::endl; count=0;}
        return mData.data() + (count * mBatchSize * samplesCommon::volume(mDims));

    }

    float* getLabels() override
    {
        return mLabels.data() + (mBatchCount * mBatchSize);
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }
    nvinfer1::Dims getDims() const override
    {
        return mDims;
    }
    nvinfer1::Dims getImageDims() 
    {
        return mDims;
    }
    int count{-1};
private:
    void readDataFile()
    {
    int channels =3;
    int height =32;
    int width =32;
    int max ;
    
    SampleINT8APIPreprocessing p;
    vector<uint8_t> fileData((mBatchSize*mMaxBatches)*channels * height * width);
//    vector<uint8_t> fileData(mMaxBatches*channels * height * width);    
    std::string magic{""};
    uint8_t *p_data;
    p_data = fileData.data();
    mData.resize((mBatchSize*mMaxBatches)*channels * height * width);
//    mData.resize(mMaxBatches*channels * height * width);
   vector<string> category;

   category.push_back("airplane");
   category.push_back("automobile");
   category.push_back("bird");
   category.push_back("cat");
   category.push_back("deer");
   category.push_back("dog");
   category.push_back("frog");
   category.push_back("horse");
   category.push_back("ship");
   category.push_back("truck");

 gLogInfo << "mBatchSize" <<mBatchSize<< std::endl;
 gLogInfo << "mMaxBatches" <<mMaxBatches<< std::endl;
 cifar10 normalize;
for(int i=0,j=0; j < mBatchSize*mMaxBatches; i++,j++)
{
   int k = j / 1000;
   
   if (j % 1000 == 0){gLogInfo<<"k = "<<k<<category[k]<<"j = "<<j ; i=0;gLogInfo<< "hhhhh2" << std::endl;}
    if(i < 9)
   {
    gLogInfo << "/home/peppa3/cifar/"+category[k]+"000"+std::to_string(i+1)+".ppm"<< std::endl;    
    std::ifstream infile("/home/peppa3/cifar/"+category[k]+"000"+std::to_string(i+1)+".ppm", std::ifstream::binary);
    
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);
   }
    else if(i < 99)
   {
    std::ifstream infile("/home/peppa3/cifar/"+category[k]+"00"+std::to_string(i+1)+".ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);
    }
    else if(i < 999)
   { 
    std::ifstream infile("/home/peppa3/cifar/"+category[k]+"0"+std::to_string(i+1)+".ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);
   }
   else
   {
    std::ifstream infile("/home/peppa3/cifar/"+category[k]+std::to_string(i+1)+".ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);

   }

    p_data += width * height * channels;


        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = j * channels *height *width + ( h * width * channels + w * channels + 2 );
                int srcIdx = j * channels *height *width + ( h * width * channels + w * channels );
                int temp = fileData[dstIdx];
                fileData[dstIdx] = fileData[srcIdx];
                fileData[srcIdx] = temp;
            }
      }

    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = j * channels *height *width + ( c * height * width + h * width + w );
                int srcIdx = j * channels *height *width + ( h * width * channels + w * channels + c );
               // mData[dstIdx] = float(fileData[srcIdx]);
                //mData[srcIdx] = (float(mData[srcIdx]) / 255.0f - mean[c]) / std[c];
               // gLogInfo << "hhhhhjjjwww" << std::endl;
                mData[dstIdx] = (float) fileData[srcIdx];
            }
        }
    }

    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = j * channels *height *width + ( c * height * width + h * width + w );

                mData[dstIdx] = (float) (mData[dstIdx] / normalize.scale -normalize.mean.at(c)) / normalize.std.at(c);
            }
        }
    }

   }


      std::cout<<"mysize is "<<mData.size()<<"success"<<std::endl;
    }

    void readLabelsFile()
    {
    vector<string> groundtruth;
   
    if (!samplesCommon::readReferenceFile("/usr/src/tensorrt/data/alexnet/groundtruth.txt", groundtruth))
    {
        gLogError << "Unable to read reference file: "  << std::endl;
            }

     for(int i = 0;i < mBatchSize*mMaxBatches;i++) mLabels.push_back(atof(groundtruth[i].c_str()));

     std::cout<<"size "<<mLabels.size()<<"   "<<mLabels[0]<<std::endl;
    }

    int mBatchSize{0};
    int mBatchCount{-1}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    Dims mDims{};
    std::vector<float> mData{};
    std::vector<float> mLabels{};
};

//********************************************************************************************************//
class PPMBatchStream : public IBatchStream
{
public:
    PPMBatchStream(int batchSize, int maxBatches,nvinfer1::Dims Dims)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{Dims}  
    {
        readDataFile();
    //    readLabelsFile();
    }

    void reset(int firstBatch) override
    {
        mBatchCount = firstBatch;
    }

    bool next() override
    {
        if (mBatchCount == mMaxBatches-1 )
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
        mBatchCount += skipCount;
    }

    float* getBatch() override
    {   
        count++;

        int num = mMaxBatches/mBatchSize;
        if( mBatchCount % num == 0) {    gLogInfo << "count is " <<mBatchCount<<std::endl; count=0;}
        return mData.data() + (count * mBatchSize * samplesCommon::volume(mDims));
    }

    float* getLabels() override
    {
        return mLabels.data() + (mBatchCount * mBatchSize);
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }
    nvinfer1::Dims getDims() const override
    {
        return mDims;
    }
    nvinfer1::Dims getImageDims() 
    {
        return mDims;
    }
    int count{-1};
private:
    void readDataFile()
    {
    int channels =3;
    int height =227;
    int width =227;
    int max ;
    
    SampleINT8APIPreprocessing p;
//    vector<uint8_t> fileData((mBatchSize*mMaxBatches)*channels * height * width);
    vector<uint8_t> fileData(mMaxBatches*channels * height * width);    
    std::string magic{""};
    uint8_t *p_data;
    p_data = fileData.data();
//    mData.resize((mBatchSize*mMaxBatches)*channels * height * width);
    mData.resize(mMaxBatches*channels * height * width);

for(int i=0; i < mMaxBatches; i++)
{
    if(i < 9)
   {

    std::ifstream infile("/home/peppa3/out2/ILSVRC2012_val_0000000"+std::to_string(i+1)+".JPEG.ppm", std::ifstream::binary);

    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);
   }
    else if(i < 99)
   {
    std::ifstream infile("/home/peppa3/out2/ILSVRC2012_val_000000"+std::to_string(i+1)+".JPEG.ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);
    }
    else if(i < 999)
   { 
    std::ifstream infile("/home/peppa3/out2/ILSVRC2012_val_00000" +std::to_string(i+1)+".JPEG.ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);
   }
   else
   {
    std::ifstream infile("/home/peppa3/out2/ILSVRC2012_val_0000" +std::to_string(i+1)+".JPEG.ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);

   }
    //gLogInfo << "hhhhh2" << std::endl;
    p_data += width * height * channels;
//    gLogInfo << "hhhhh3" << std::endl;

    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = i * channels *height *width + ( c * height * width + h * width + w );
                int srcIdx = i * channels *height *width + ( h * width * channels + w * channels + c );
   // gLogInfo << "hhhhh4" << std::endl;
                mData[dstIdx] = float(fileData[srcIdx]);
            }
        }
    }
  // gLogInfo << "hhhhhjjjwww" << std::endl;
   }


      std::cout<<"mysize is "<<mData.size()<<"success"<<std::endl;
    }

    void readLabelsFile()
    {
    vector<string> groundtruth;
   
    if (!samplesCommon::readReferenceFile("/usr/src/tensorrt/data/alexnet/groundtruth.txt", groundtruth))
    {
        gLogError << "Unable to read reference file: "  << std::endl;
            }

     for(int i = 0;i < mBatchSize*mMaxBatches;i++) mLabels.push_back(atof(groundtruth[i].c_str()));

     std::cout<<"size "<<mLabels.size()<<"   "<<mLabels[0]<<std::endl;
    }

    int mBatchSize{0};
    int mBatchCount{-1}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    Dims mDims{};
    std::vector<float> mData{};
    std::vector<float> mLabels{};
};
class PPM224BatchStream : public IBatchStream
{
public:
    PPM224BatchStream(int batchSize, int maxBatches,nvinfer1::Dims Dims)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{Dims}  
    {
        readDataFile();
    //    readLabelsFile();
    }

    void reset(int firstBatch) override
    {
        mBatchCount = firstBatch;
    }

    bool next() override
    {
        if (mBatchCount == mMaxBatches-1 )
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
        mBatchCount += skipCount;
    }

    float* getBatch() override
    {   
        return mData.data() + (mBatchCount * mBatchSize * samplesCommon::volume(mDims));
    }

    float* getLabels() override
    {
        return mLabels.data() + (mBatchCount * mBatchSize);
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }
    nvinfer1::Dims getDims() const override
    {
        return mDims;
    }
    nvinfer1::Dims getImageDims() 
    {
        return mDims;
    }
    int count{-1};
private:
    void readDataFile()
    {
    int channels =3;
    int srcheight = 256 ;
    int srcwidth = 256 ;

    int dstheight =224;
    int dstwidth =224;
    std::string magic{""};
    int max;
    int starth = srcheight / 2 - (dstheight + dstheight % 2) / 2;
    int startw = srcwidth / 2 - (dstwidth + dstwidth % 2) / 2;

    vggPreprocessing p;
    vector<uint8_t> fileData(mBatchSize*mMaxBatches*channels * dstheight * dstwidth);


    vector<uint8_t> srcData(mBatchSize*mMaxBatches*channels * srcheight * srcwidth);
    uint8_t *p_data;
    p_data = srcData.data();


    mData.resize(mBatchSize*mMaxBatches*channels * dstheight * dstwidth);

    std::vector<float> rgbdata{};
    rgbdata.resize(mMaxBatches*mBatchSize*channels * dstheight * dstwidth);
   

    std::vector<float> means{};
    means.resize(channels * dstheight * dstwidth);

//    std::ifstream input("/home/peppa3/means.txt"); 
FILE *fp = fopen("/home/peppa3/means.txt","r");
     float temp;
for (int i=0 ;i<channels * dstheight * dstwidth;i++)
{
 fscanf(fp,"%f\n",&temp);
 means[i]=temp;
 if(i <5) {gLogInfo << "mean is" <<temp<< std::endl;}
}
fclose(fp);
/*     string s;
    while(getline(input,s)){ 
     stringstream str(s);
     float temp;
     str>>temp;
//     means[i++]=std::atof(s.c_str());
     means[i++]=temp;
     if(i <5) {gLogInfo << "mean is" <<temp<< std::endl;}
    } 
    input.close();
   gLogInfo << "count is" <<i<< std::endl;*/

for(int i=0; i < mBatchSize*mMaxBatches; i++)
{
    if(i < 9)
   {

    std::ifstream infile("/home/peppa3/image/ILSVRC2012_val_0000000"+std::to_string(i+1)+".JPEG.ppm", std::ifstream::binary);

    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> srcwidth >> srcheight >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), srcheight * srcwidth * channels);
   }
    else if(i < 99)
   {
    std::ifstream infile("/home/peppa3/image/ILSVRC2012_val_000000"+std::to_string(i+1)+".JPEG.ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> srcwidth >> srcheight >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), srcheight * srcwidth * channels);
    }
    else if(i < 999)
   { 
    std::ifstream infile("/home/peppa3/image/ILSVRC2012_val_00000" +std::to_string(i+1)+".JPEG.ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> srcwidth >> srcheight >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), srcheight * srcwidth * channels);
   }
   else
   {
    std::ifstream infile("/home/peppa3/image/ILSVRC2012_val_0000" +std::to_string(i+1)+".JPEG.ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> srcwidth >> srcheight >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), srcheight * srcwidth * channels);

   }
    p_data += srcheight * srcwidth * channels;

for (int c = 0; c < channels; ++c)
{
    for (int h = starth; h < starth + dstheight; ++h)
    {
        for (int w = startw; w < startw + dstwidth; ++w)
        {
            int srcIdx = i * channels *srcheight * srcwidth + h * srcwidth * channels + w * channels + c;
            int dstIdx = i * channels *dstheight *dstwidth + ( h - starth ) * dstwidth * channels + ( w - startw ) * channels + c;
            
             fileData[dstIdx] = srcData[srcIdx];
        }
    }
}

for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < dstheight; ++h)
        {
            for (int w = 0; w < dstwidth; ++w)
            {
                int dstIdx = i * channels *dstheight *dstwidth + ( c * dstheight * dstwidth + h * dstwidth + w );
                int srcIdx = i * channels *dstheight *dstwidth + ( h * dstwidth * channels + w * channels + c );
                rgbdata[dstIdx] = float(fileData[srcIdx]);
            }
        }
    }
}

for(int j=0;j<mBatchSize*mMaxBatches;j++){
    for(int i=0;i<dstheight*dstwidth;i++)
    {
         mData[j*channels*dstheight*dstwidth+i] = rgbdata[j*channels*dstheight*dstwidth+2*dstheight*dstwidth+i]-means[i];
    }

    for(int i=0;i<dstheight*dstwidth;i++)
    {
          mData[j*channels*dstheight*dstwidth+dstheight*dstwidth+i] =rgbdata[j*channels*dstheight*dstwidth+dstheight*dstwidth+i]-means[dstheight*dstwidth+i];
    }

    for(int i=0;i<dstheight*dstwidth;i++)
    {
         mData[j*channels*dstheight*dstwidth+2*dstheight*dstwidth+i] = rgbdata[j*channels*dstheight*dstwidth+i]-means[2*dstheight*dstwidth+i];
    }

    }
}
    void readLabelsFile()
    {
    vector<string> groundtruth;
   
    if (!samplesCommon::readReferenceFile("/usr/src/tensorrt/data/alexnet/groundtruth.txt", groundtruth))
    {
        gLogError << "Unable to read reference file: "  << std::endl;
            }

     for(int i = 0;i < mBatchSize*mMaxBatches;i++) mLabels.push_back(atof(groundtruth[i].c_str()));

     std::cout<<"size "<<mLabels.size()<<"   "<<mLabels[0]<<std::endl;
    }

    int mBatchSize{0};
    int mBatchCount{-1}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    Dims mDims{};
    std::vector<float> mData{};
    std::vector<float> mLabels{};
};
class PPMBatchStream1 : public IBatchStream
{
public:
    PPMBatchStream1(int batchSize, int maxBatches,nvinfer1::Dims Dims)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{Dims}  
    {
        readDataFile();
       // readLabelsFile();
    }

    void reset(int firstBatch) override
    {
        mBatchCount = firstBatch;
    }

    bool next() override
    {
        if (mBatchCount == mMaxBatches-1 )
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
        mBatchCount += skipCount;
    }

    float* getBatch() override
    {
 std::cout<<"this is "<<mBatchCount<<"   "<< (mBatchCount * mBatchSize * samplesCommon::volume(mDims))<<std::endl;
        return mData.data() + (mBatchCount * mBatchSize * samplesCommon::volume(mDims));
    }

    float* getLabels() override
    {
 std::cout<<"Label is "<<mBatchCount<<"   "<< mBatchCount * mBatchSize <<std::endl;
        return mLabels.data() + (mBatchCount * mBatchSize);
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }
    nvinfer1::Dims getDims() const override
    {
        return mDims;
    }
    nvinfer1::Dims getImageDims() 
    {
        return mDims;
    }

private:
    void readDataFile()
    {
    int channels =3;
//    int height =227;
//    int width =227;
int height =224;
int width =224;
    int max ;
    
    SampleINT8APIPreprocessing p;
    vector<uint8_t> fileData((mBatchSize*mMaxBatches)*channels * height * width);

    std::string magic{""};
    uint8_t *p_data;
    p_data = fileData.data();
    mData.resize((mBatchSize*mMaxBatches)*channels * height * width);

for(int i = 0; i < mBatchSize*mMaxBatches; i++){

      if(i < 9)
   {
    std::ifstream infile("/usr/src/tensorrt/data/googlenet/batch/batch/ILSVRC2012_val_0000000"+std::to_string(i+1)+".JPEG.ppm", std::ifstream::binary);

    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);
   }
    else if(i < 99)
   {
    std::ifstream infile("/usr/src/tensorrt/data/googlenet/batch/batch/ILSVRC2012_val_000000"+std::to_string(i+1)+".JPEG.ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);
    }
    else if(i < 999)
   { 
    std::ifstream infile("/usr/src/tensorrt/data/googlenet/batch/batch/ILSVRC2012_val_00000" +std::to_string(i+1)+".JPEG.ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);
   }
   else
   {
    std::ifstream infile("/usr/src/tensorrt/data/googlenet/batch/batch/ILSVRC2012_val_0000" +std::to_string(i+1)+".JPEG.ppm", std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(p_data), width * height * channels);

   }

    p_data += width * height * channels;

    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = i * channels *height *width + ( c * height * width + h * width + w );
                int srcIdx = i * channels *height *width + ( h * width * channels + w * channels + c );

                     mData[dstIdx] = float(fileData[srcIdx]);
            }
        }
    }
  
   }
      std::cout<<"mysize is "<<mData.size()<<"success"<<std::endl;
    }

    void readLabelsFile()
    {
    vector<string> groundtruth;
   
    if (!samplesCommon::readReferenceFile("/usr/src/tensorrt/data/alexnet/groundtruth.txt", groundtruth))
    {
        gLogError << "Unable to read reference file: "  << std::endl;
        
    }

     for(int i = 0;i < mBatchSize*mMaxBatches;i++) mLabels.push_back(atof(groundtruth[i].c_str()));

     std::cout<<"size "<<mLabels.size()<<"   "<<mLabels[0]<<std::endl;
    }

    int mBatchSize{0};
    int mBatchCount{-1}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    Dims mDims{};
    std::vector<float> mData{};
    std::vector<float> mLabels{};
};
class BatchStream : public IBatchStream
{
public:
    BatchStream(
        int batchSize, int maxBatches, std::string prefix, std::string suffix, std::vector<std::string> directories)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mPrefix(prefix)
        , mSuffix(suffix)
        , mDataDir(directories)
    {
        FILE* file = fopen(locateFile(mPrefix + std::string("0") + mSuffix, mDataDir).c_str(), "rb");
        assert(file != nullptr);
        int d[4];
        size_t readSize = fread(d, sizeof(int), 4, file);
        assert(readSize == 4);
        mDims.nbDims = 4;  // The number of dimensions.
        mDims.d[0] = d[0]; // Batch Size
        mDims.d[1] = d[1]; // Channels
        mDims.d[2] = d[2]; // Height
        mDims.d[3] = d[3]; // Width
        assert(mDims.d[0] > 0 && mDims.d[1] > 0 && mDims.d[2] > 0 && mDims.d[3] > 0);
        fclose(file);

        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        mFileLabels.resize(mDims.d[0], 0);
        reset(0);
    }

    BatchStream(int batchSize, int maxBatches, std::string prefix, std::vector<std::string> directories)
        : BatchStream(batchSize, maxBatches, prefix, ".batch", directories)
    {
    }

    BatchStream(
        int batchSize, int maxBatches, nvinfer1::Dims dims, std::string listFile, std::vector<std::string> directories)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mDims(dims)
        , mListFile(listFile)
        , mDataDir(directories)
    {
        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        mFileLabels.resize(mDims.d[0], 0);
        reset(0);
    }

    // Resets data members
    void reset(int firstBatch) override
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.d[0];
        skip(firstBatch);
    }

    // Advance to next batch and return true, or return false if there is no batch left.
    bool next() override
    {
        if (mBatchCount == mMaxBatches)
        {
            return false;
        }

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.d[0]);
            if (mFileBatchPos == mDims.d[0] && !update())
            {
                return false;
            }

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.d[0] - mFileBatchPos);
            std::copy_n(
                getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
            std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
        }
        mBatchCount++;
        return true;
    }

    // Skips the batches
    void skip(int skipCount) override
    {
        if (mBatchSize >= mDims.d[0] && mBatchSize % mDims.d[0] == 0 && mFileBatchPos == mDims.d[0])
        {
            mFileCount += skipCount * mBatchSize / mDims.d[0];
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
        {
            next();
        }
        mBatchCount = x;
    }

    float* getBatch() override
    {
        return mBatch.data();
    }

    float* getLabels() override
    {
        return mLabels.data();
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override
    {
        return mDims;
    }

private:
    float* getFileBatch()
    {
        return mFileBatch.data();
    }

    float* getFileLabels()
    {
        return mFileLabels.data();
    }

    bool update()
    {
        if (mListFile.empty()) //一个batch的图片存在一个文件里面
        {
            std::string inputFileName = locateFile(mPrefix + std::to_string(mFileCount++) + mSuffix, mDataDir);
            FILE* file = fopen(inputFileName.c_str(), "rb");
            if (!file)
            {
                return false;
            }

            int d[4];
            size_t readSize = fread(d, sizeof(int), 4, file);
            assert(readSize == 4);
            assert(mDims.d[0] == d[0] && mDims.d[1] == d[1] && mDims.d[2] == d[2] && mDims.d[3] == d[3]);
            size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.d[0] * mImageSize, file);
            assert(readInputCount == size_t(mDims.d[0] * mImageSize));
            size_t readLabelCount = fread(getFileLabels(), sizeof(float), mDims.d[0], file);
            assert(readLabelCount == 0 || readLabelCount == size_t(mDims.d[0]));

            fclose(file);
        }
        else
        {
            std::vector<std::string> fNames;
            std::ifstream file(locateFile(mListFile, mDataDir), std::ios::binary);
            if (!file)
            {
                return false;
            }

            gLogInfo << "Batch #" << mFileCount << std::endl;
            file.seekg(((mBatchCount * mBatchSize)) * 7);

            for (int i = 1; i <= mBatchSize; i++)
            {
                std::string sName;
                std::getline(file, sName);
                sName = sName + ".ppm";
                gLogInfo << "Calibrating with file " << sName << std::endl;
                fNames.emplace_back(sName);
            }

            mFileCount++;

            const int imageC = 3;//？？
            const int imageH = 300;
            const int imageW = 300;
            std::vector<samplesCommon::PPM<imageC, imageH, imageW>> ppms(fNames.size());
            for (uint32_t i = 0; i < fNames.size(); ++i)
            {
                readPPMFile(locateFile(fNames[i], mDataDir), ppms[i]);
            }

            std::vector<float> data(samplesCommon::volume(mDims));
            const float scale = 2.0 / 255.0;
            const float bias = 1.0;
            long int volChl = mDims.d[2] * mDims.d[3];

            // Normalize input data
            for (int i = 0, volImg = mDims.d[1] * mDims.d[2] * mDims.d[3]; i < mBatchSize; ++i)
            {
                for (int c = 0; c < mDims.d[1]; ++c)
                {
                    for (int j = 0; j < volChl; ++j)
                    {
                        data[i * volImg + c * volChl + j] = scale * float(ppms[i].buffer[j * mDims.d[1] + c]) - bias;
                    }
                }
            }

            std::copy_n(data.data(), mDims.d[0] * mImageSize, getFileBatch());//读取一个batch的数据放到mFileData里面
            
        }

        mFileBatchPos = 0;
        return true;
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    int mFileCount{0};
    int mFileBatchPos{0};
    int mImageSize{0};
    std::vector<float> mBatch;         //!< Data for the batch
    std::vector<float> mLabels;        //!< Labels for the batch
    std::vector<float> mFileBatch;     //!< List of image files 
    std::vector<float> mFileLabels;    //!< List of label files
    std::string mPrefix;               //!< Batch file name prefix
    std::string mSuffix;               //!< Batch file name suffix
    nvinfer1::Dims mDims;              //!< Input dimensions
    std::string mListFile;             //!< File name of the list of image names
    std::vector<std::string> mDataDir; //!< Directories where the files can be found
};

#endif
