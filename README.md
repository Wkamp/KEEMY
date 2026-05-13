# KEEMY

KEEMY (Kentucky Efficient Error Modeler) is a fast blind image denoiser developed at the University of Kentucky’s KAOS research lab. It is an improved version of KEMY, sharing the same self-derived statistical error model while improving runtime and denoising quality through more efficient patch correspondence and reconstruction.

## Performance and Results

Compared to KEMY, KEEMY achieves up to 5.0× (400%) faster multithreaded performance and 7.8× (679%) faster single-threaded performance. Single-threaded KEEMY is also 1.37× (37%) faster than multithreaded KEMY.

Below is an example comparison from my project writeup.

<p align="center">
  <img src="images/denoiser_comparison.png" width="1000">
</p>

## More Information

The KEEMY website writeup provides implementation details, benchmarking results, image comparisons, and a full explanation of the denoising pipeline, along with background information for those without prior experience in image processing.
- [KEEMY writeup](https://eiron.xyz/keemy/)
- [KEMY project](https://aggregate.org/DIT/KEMY/) (original method)

