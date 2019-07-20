# Co-Training Implementation for Objectivity Detection with R
Data analytics project in R to create a predictive model for the detection of objectivity in sports articles, based on co-training.

<a href="https://github.com/ansegura7/ML_ObjectivityDetection/blob/master/paper/Co-Training_Implementation_for_Objectivity_Detection.pdf" target="_blank">Paper</a>

## Abstract
In the current era in which we live, many sports articles are published daily on the Internet, by various authors, which are often written objectively, on other occasions subjectively, which may not please the reader or change your perception of the facts. In the present work, we perform a detection analysis of objectivity to a set of 1000 sports articles, previously labeled using the Mechanical Turk tool from Amazon. For this, we conducted 2 experiments in which the predictive ability of using trained statistical models with supervised versus trained learning with semi-supervised learning was compared. The fact of learning from the original file tagged by Amazon Mechanical Turk and one generated with the TMG+ algorithm based on TF-IDF was also evaluated. The results obtained were very encouraging, since the SL approach generated better results for the original tagged file (precision close to 82.9%), while the SSL approach using Co-Training, generated better results for the dataset created with the own algorithm, with accuracy close to 74.4% using 50% of the data tagged for SSL.

## Data

## Technologies and Techniques
- R 3.5.1 x64
- RStudio - Version 1.1.383
- Supervised Learning (SL)
- Co-training

## Contributing and Feedback
Any kind of feedback/criticism would be greatly appreciated (algorithm design, documentation, improvement ideas, spelling mistakes, etc...).

## Authors
- Created by Segura Tinoco, Andrés and <a href="https://github.com/vladcuevas" target="_blank">Cuevas Saavedra, Vladimir</a>
- Created on July, 2018

## License
This project is licensed under the terms of the MIT license.
