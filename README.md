# monophonic-pitch-detection
For a project I was asked to help with monophonic pitch detection (for piano), so in this repository I use and compare different algorithms and libraries to achieve the task.  

I started out with _aubio_ which has a nice variant of YIN called YinFFT. It is fast and >96% accurate on slow pieces (~1 note per second, which is realistic for this project). The regular YIN used in combination with a low pass filter gets almost the same accuracy, but is about 15x slower.

For onset detection _aubio_ has several methods that perform well: simple `local energy`, high frequency content `hfc` and the popular `specflux`. After some tuning `specflux` performs best in my use case with ~95-97%.
The _madmom_ library additionally provides `superflux`, which outperforms `specflux` by another 2%, but it is not optimized for real time detection, making it unnecessary slow. _madmom_ also provides pre-trained Neural Nets, but they were not trained specifically for piano and in my tests they performed worse than `superflux`.