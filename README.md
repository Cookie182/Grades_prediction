# University DBMS Project to help store and predict grades
### Using Python to generate mock data depending on each subjects structure and making models to help get them ready to start predicting 2 aspescts of a students per subject.
* The probability of the student passing as they complete each assessment
* The predicted end overall grade of that subject as students complete aech assessment

### There are four files that contain my way of trying to figure out ways to generate mock data and to find the best models for predicting grades. Training data was generated with using a range of numbers (depending on a particular subjects structure) that has a Gaussian weight distribution attached to it 
* ### [Testing Models](https://github.com/Cookie182/Grades_prediction/blob/main/Testing%20Models.ipynb "Main playground for data generation, models and neural networks") - *Playground to find ways to generate mock data and evaluate different models, even an attempt at Sequential Neural Networks to see how they perform compared to traditional machine learning algorithms*

* ### [Models](https://github.com/Cookie182/Grades_prediction/blob/main/Models.ipynb "Making a function to generate data and models depending on subject structure to use in demo") - *The second playground where a function was made that can be used for the demo of the system. The function is will output 2 models (1 Classifier and 1 Regressor) to generate 2 kinds of predictions based on randomly generated mock data that was generated based on a subjects structure*

* ### [Old demo](https://github.com/Cookie182/Grades_prediction/blob/main/old%20demo.py "First go at writing a script for a CLI demo of our system") - *Using the code from the previous Jupyter notebooks and the code that was made there to be used in making a CLI demo of our system*

* ### [Demo](https://github.com/Cookie182/Grades_prediction/blob/main/demo.py "Final version of the CLI demo") - *Final version of the CLI demo using the code that was written previously in all the files, except it is slightly more organised here. More functionality options were also added.*
