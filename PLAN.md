I need a pipeline to convert mp4 files into sequences of 16 frames

these frames will be default be much larger than i want. i need to convert them from their resolution down to around 128 by 128

ideally i can also convert from rgb 3 channel (also harder to learn since rgb can tae way more possible values) into just one channel with some number of discrete values (4 or 16 or so depending on the game chosen).

I then need a way to learn a neural net, and a visualizer which converts these 4 or 16-range ints into rgb pixel values.

for this i should just create a map from RGB value to int pixel value, and for data coming in just use kmeans to convert

I can just download mp4 videos of gameplay: eliminates need to play it out myself or deal with aids emulator shit

skill is then in

1. model construction
2. creating the video display, need to make sure I get it right
    1. I can verify this by using the 4-bit quantization then making the displayer and making sure the video is coherent

    