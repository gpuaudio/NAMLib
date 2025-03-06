# Neural Amp Modeler Client Library
This is the client library for the GPU Neural Amp Modeler, which takes care of
loading the GPU Audio engine and the `nam_processor` and offers a lightweight
interface to process audio data either synchronously or asynchronously with
double buffering. Furthermore a simple command line application is included,
which processes a `*.wav` file with the `nam_processor`.

## NeuralAmpModelerInterface
Interface for the client library and public include for projects using the library.

## GPUCreate
Function to create an instance of the client library, i.e., GPUNeuralAmpModelerWavenet or GPUNeuralAmpModelerLSTM

## GPUNeuralAmpModeler
Actual implementations of the NeuralAmpModelerInterface.

# GPUNeuralAmpModelerTests
Unit tests to check correctness of processor and library and to illustrate how to use the
library to process data.

## nam_process_file
Command line application, to processes an audio file with the `nam_processor` in a user-defined
model configuration.
