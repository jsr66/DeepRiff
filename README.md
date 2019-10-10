## DeepRiff Overview

DeepRiff is an AI music idea generator of jazz, blues, ragtime, and classical music, which I built 
during my time as a fellow in the Insight fellowship program. The purpose is to generate new musical ideas,
or riffs, in the style of a user-chosen artist or song. The user makes a selection, waits a few minutes while DeepRiff
composes, and is then taken to a page where they can download a midi file and pdf sheet music of the new composition. 
DeepRiff generates both "beginner" and "advanced" compositions. Code for training an lstm that generates beginner compositions, and the code to generate these compositions from the trained model, are contained in the folder "beginner." Code for training an AWD LSTM that generates advanced compositions, and the code to generate these compositions from the trained model, are contained in the folder "advanced." 
