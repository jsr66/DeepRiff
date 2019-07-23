#Introducing DeepRiff
DeepRiff (http://deepriff.com/) is an AI music idea generator of jazz, blues, ragtime, and classical music, which I built 
during my time as a fellow in the Insight fellowship program. The purpose is to generate new musical ideas,
or riffs, in the style of a user-chosen artist or song. The user makes a selection, waits a few minutes while DeepRiff
composes, and is then taken to a page where they can download a midi file and pdf sheet music of the new composition. 

DeepRiff generates both "beginner" and "advanced" compositions. The multi-layer LSTM models used to generate the beginner compositions were
trained using code adapted from https://github.com/Skuldur/Classical-Piano-Composer. The models used to generate the 
advanced compositions were trained using code adapted from https://github.com/mcleavey/musical-neural-net, and were based on 
more sophisticated AWD LSTM models. 
