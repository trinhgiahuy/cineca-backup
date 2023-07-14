#include <stdio.h>

int main() {
  // FIX ME: assign values to these variables
  // to prints the requested properties of the 
  // currently active GPU.

  int deviceId;
  int computeCapabilityMajor;
  int computeCapabilityMinor;
  int multiProcessorCount;
  int warpSize;

  printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
}
