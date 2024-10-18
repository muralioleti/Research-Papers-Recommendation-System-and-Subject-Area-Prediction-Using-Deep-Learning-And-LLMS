# _Research Papers Recommendation System and Subject Area Prediction Using Deep Learning And LLMS_

# _Overview_
_This repository hosts a machine learning project that encompasses two key functionalities: a research papers recommendation system and subject area prediction. The goal of the project is to provide users with tailored recommendations based on their preferences and to predict the subject area of research papers._

# _Key Features_

- _LSTM model for text classification_
- _Text vectorization using Keras's TextVectorization layer_
- _Multi-class classification with evaluation metrics (precision, recall, F1-score)_
- _Error handling for multi-class AUC-ROC computation_

# _Set-up Instructions_
_**Install dependencies:**_ _Ensure you have the required libraries installed:_

- _`pip install -r requirements.txt`_

_**Load the pre-trained model and vectorizer:**_ _The model and text vectorizer configurations are loaded from pre-saved files in the project:_

- _**model :**_ _`/path/to/model`_
- **_vocabulary :_** _`/path/to/vocab.pkl`_
- _**text_vectorizer_config :**_ _`/path/to/text_vectorizer_config.pkl`_

# _Usage_

- _**Running the prediction:** After loading the model and vectorizer, run predictions using your test dataset._
- _**Evaluating the Model:** The script calculates precision, recall, F1-score, and AUC-ROC, handling multi-class cases:_
_`precision, recall, f1_score, auc_roc = evaluate_model(y_test, y_pred)`_

# _Deep Learning Model:_
_Implementing a deep learning model to capture complex patterns implementing MLP to predict subject area of a paper._

# _Subject Area Prediction:_
_Text Classification: Utilizing natural language processing techniques for classifying research papers into subject areas._

# _Key Metrics:_

- _**Precision:** Measures the accuracy of the positive predictions._
- _**Recall:** Measures how well the model retrieves relevant instances._
- _**F1-Score:** Harmonic mean of precision and recall._
- _**AUC-ROC:** Area under the ROC curve for multi-class classification._

# _Recommendation System:_
_Leveraging sentence transformers using sentence embedding with cosine similarity techniques to recommend research papers based on user preferences and similarities to other vectors._

- _`arxiv_data.drop(columns = ["terms","abstracts"], inplace = True)`_

- _`arxiv_data.drop_duplicates(inplace= True)                                                                                         
   arxiv_data.reset_index(drop= True,inplace = True)`_

- _`pd.set_option('display.max_colwidth', None)                                                                              
 arxiv_data`_

# _Results:_
_Research Papers Recommendation Recommending top K papers. The deep learning model improved recommendations, yielding an accuracy of 99%. Subject Area Prediction._
