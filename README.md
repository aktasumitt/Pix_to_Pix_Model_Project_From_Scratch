# Pix2Pix Model WITH UT-Zappos50K Dataset

## Introduction:
-In this project, I aimed to train a Pix2Pix model architecture for image to image translation.
-The aim of the training is to transform 1-dimensional gray shoe sketches into 3-dimensional RGB real shoe images.

## Models:
- For training, I utilized the Pix2pix model, which I coded from scratch.
- This model is essentially like a conditional GAN, comprising a generator and a discriminator. However, the generator differs in that it's based on a U-net architecture. 
- The U-net operates similar to an encoder-decoder-based autoencoder.
- It takes an input, unsamples it to a certain feature map size, and then upsamples it back to the input size.
- An important feature of U-net is that, while passing from the encoder to the decoder, the same-sized connections from the encoder are sequentially concatenated to the input of the decoder
- The discriminator performs downsampling to image dimensions of (batch_size,1,30,30) for prediction.

- Pix2Pix Model link: https://arxiv.org/abs/1611.07004


