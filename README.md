# monophonic-pitch-detection
For a project I was asked to help with monophonic pitch detection (for piano), so in this repository I use and compare different algorithms and libraries to achieve the task.  

I started out with _aubio_ which has a nice variant of YIN caled YinFFT. It is fast and >96% accurate (notice: benchmark contains only 129 annotated pitches so far). The regular YIN used in combination with a low pass filter gets the same results, but is about 15x slower.

For Onset _aubio_ has several methods that perform well: simple `local energy`, high frequency content `hfc` and the popular `specflux`  
The _madmom_ library additionally provides `superflux` as well as several pre-trained Neural Nets, but so far I haven't figured the correct parameters to match the performance of the _aubio_ library.