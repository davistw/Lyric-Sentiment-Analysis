******************** OVERVIEW ********************

This readme file contains instructions for setting up and running two entities: an interactive visualization demo
and a lyric sentiment analysis demo. Description, installation, and execution sections are provided for both demos 
and are shown below.



******************** INTERACTIVE VISUALIZATION DEMO ********************


DESCRIPTION:

     Our interactive visualization is a user interface that allows users to explore the results of our music 
lyric sentiment analysis. This demo version utilizes a ~15,000 record subset of the full data that was produced 
during our experiments and analysis (see our report for a breakdown on the nature of this sample). This
demo interface is delivered through a locally hosted web-page that allows users to choose from several 
options concerning which data subsets/metrics to display: experiment type, artist, emotion, and top scores.

    The user interface is in the form of an interactive scatterplot that allows users to scale the magnification
of the plot canvas to view data points in more detail. Click and drag functionality for the plot canvas is also
included. Data points on the plot are color-coded according to their relevant sentiment classifications. A 
corresponding legend is also included. A "reset button" is included in the lower left-hand corner that returns 
the plot view to its original position and scale. 

INSTALLATION:

     With regard to installation procedures for the interactive visualization, it is important that you have Python
3.7 or greater installed, as well as a functioning web browser such as Mozila Firefox or Google Chrome. Once that
is established, the only step that is then required is to download and unzip the "team180final.zip" file onto your 
local system.

EXECUTION:

     To run the interactive visualization demo, please perform the following steps in numerical order:

   1) Open the command prompt on your local system.

   2) Set the current directory to the "Interactive Visualization" folder located inside the "CODE" folder of the 
      team180final file that was unzipped during installation. This can be accomplished by typing the command "cd" 
      followed by the folder path, then pressing the enter key:

                 (i.e.): cd C:\Users\User_0\Documents\team180final\CODE\Interactive Visualization
   
   3) Start a simple, local Python server by typing the following command into the command prompt: 

                  python -m http.server 8000 ("8000" represents the recommended port number, 
                                               but other numbers can be used)

   4) Open a web browser such as Mozilla Firefox or Google Chrome and type the following url into the address bar:

                  http://localhost:8000/visualization interface.html (8000 can be replaced with whichever port #
                                                                      you chose when setting up the server in step 3)

      You will now be able to explore the interactive user interface.




******************** LYRIC SENTIMENT ANALYSIS DEMO ********************


DESCRIPTION:

     Our lyric sentiment analysis is a python program that assigns sentiment to music lyrics using classification via
the VADER and spaCy Python libraries, as well as K-Means clustering. This demo version creates and runs a small 
(5,000 record) randomized sample through our analysis from the much larger data used in our original experiments.

     Our VADER library-based analysis classifies the song lyrics into three categories: Positive, Negative, and Neutral.
Our spaCy library analysis and K-Means clustering analysis classifies the song lyrics into four emotional categories:
Happy, Sad, Angry, and Relaxed.


INSTALLATION:

   1) Much like our visualization, please ensure that Python 3.7 or greater is installed on your local machine.
   
   2) In addition to standard Python libraries, please install the following libraries and library components
      by using pip and the following commands:
         
         $ pip install spacy
         $ python -m spacy download en_core_web_sm
         $ pip install langid
         $ pip install vaderSentiment
         $ pip install sklearn

   3) Download the following datasets:

        -- lyrics_features.csv and songs_dataset.csv from https://www.kaggle.com/detkov/lyrics-dataset 
           (under the Data tab scroll down to Data Sources box, click on the data source and use the 
           download icon in the data preview box to start the download)
           
           *Note: You may be required to login/create a kaggle.com account in order to complete the down-
                  load. The process is very simple and requires only a working email address, 7 character
                  minimum password, and completion of a simple captcha puzzle/email verification.

        -- Ratings_Warriner_et_al.csv from http://crr.ugent.be/archives/1003 (Click the hyperlink in "You 
           find the affective ratings here." to start the download)

   4) Extract file sample.py from the Lyrics Sentiment Analysis folder inside the CODE folder of the team180final 
      file which was unzipped during the interactive visualization installation steps.
        
        -- The code in this file will run lyric sentiment analysis on 5000 randomly selected songs and should 
           run in approximately 10 to 15 minutes.
     
        -- The code to run the entire data set is available upon request, it was not included with the submission 
           due to its excessively long execution time.

   5) Ensure the three previosly dowloaded data sets and sample.py script are in the same local directory.


EXECUTION:

   1) Open command line and navigate to the local directory containing data set files and sample.py script.

   2) Execute sample.py script by typing the following command: python sample.py

   3) Upon completion of the script execution, the output file containing 5000 songs and assigned sentiment labels will 
      be available in the local directory from which the code was run with the name "processed_lyrics_sample.csv". 
      Cluster_Label column contains the final result of lyrics sentiment analysis using clustering on valance and arousal 
      scores of songs.

  



   