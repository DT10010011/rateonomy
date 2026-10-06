Setup instructions

To run the test file, use the command "python3 test.py [filename]" with your downloaded farmland image file (don't include the brackets).
The easiest way to do this is to open VSCode, have all of the files in the same folder, and type the command in the VSCode terminal.
Also, you first need to obtain a Gemini API key from Google AI Studio.
Google offers a free tier of the Gemini API, so you won't be charged for running this code.
However, the number of requests you can send to Gemini is limited on the free tier, and when the limit
is reached you'll have to wait to run this file again.

The test pipeline currently uses Gemini Flash 2.5. From my observation, outputted scores tend to vary by +-1 with the same image, so a future step
in the implementation could be to send N prompts and average the scores for more consistent results.
