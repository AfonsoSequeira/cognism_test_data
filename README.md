# cognism_test_data
For Task 1, the method handles most company names well, apart from a few exceptions where the algorithm chooses the wrong location. For example, "ARC International North America Inc", where it chooses "North" as the location.

For Task 2, I added another method called predict_legal_identifiers(self, comp_name), which uses all the company names in company.txt to train a Naive Bayes Classifier to predict whether a comp_name contains a legal identifier or not. I could not get this method to work successfully since it could only predict 24% of the companies with legal identifiers. 
The classifier did get an overall accuracy of 83%, but this was mainly due to the class imbalance, since 76% of company names did not contain a legal identifier. To train this classifier, I used a slightly longer list of legal identifiers which I found manually. I understand that this was probably not the most correct way but I found no solution around it. I have attached this file (legal_train.txt) as well.