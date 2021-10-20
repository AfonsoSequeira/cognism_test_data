import re  
import os
import csv 

class NameComponents():
    def __init__(self):
        #open legal.txt and extract all legal identifiers
        self.legal_ids = []
        with open("legal.txt", 'r') as file:
            for line in file:
                self.legal_ids.append(line.rstrip().lower())
        
        #open locations.tsv and extract locations
        self.location_ids = []
        with open("locations.tsv", 'r', encoding="utf8") as file:
            rd = csv.reader(file, delimiter="\t", quotechar='"')
            for line in rd:
                #if location is state, make all letters upper case, otherwise capitalise 1st
                if line[1] == "state":
                    self.location_ids.append(line[0].upper())
                else:
                    self.location_ids.append(line[0].capitalize())
    
    
    def get_name_components(self, comp_name):
        
        my_dict = {}
        #save raw
        my_dict["raw"] = comp_name
        #strip input of all non-aphanumeric carachters and lower case
        comp_name = re.sub(r'\W+', ' ', comp_name)#.lower()
        comp_name = " " + comp_name + " "
            
        #iterate through list_ids and see if any are in comp_name
        for legal in self.legal_ids:
            if " " + legal + " " in comp_name.lower():
                my_dict["legal"]  = legal
        
        
        #find index of beginning of legal substrings, remove legal part from comp_name
        if "legal" in my_dict.keys():  
            idx = comp_name.lower().index(my_dict["legal"])
            comp_name = comp_name[0: idx:] + comp_name[idx +
                                len(my_dict["legal"]) + 1::]
        
        #find locations contained in string
        temp = []
        for location in self.location_ids:
            if " " + location + " " in comp_name:
                temp.append(location)
        
        #set location to largest string contained in comp_name
        #e.g choose "North Carolina" over "North"
        if len(temp) > 0:
            my_dict["location"]  = max(temp, key=len)
        
        #remove location part from comp_name
        if "location" in my_dict.keys():
            idx2 = comp_name.index(my_dict["location"])
            comp_name = comp_name[0: idx2:] + comp_name[idx2 +
                                len(my_dict["location"]) + 1::]
        
        #set base_name to remainder of comp_name
        my_dict["base_name"]  = comp_name.rstrip()
        
        return my_dict
    
    
    """
    Predict_legal_identifiers uses company list to train a Naive Bayes classifier
    to predict whether a company name contains a legal identifier or not. 
    """
    def predict_legal_identifiers(self, comp_name):
        
        #use a slighly more complete version of legal_ids from legal_train
        legal_id = []
        with open("legal_train.txt", 'r') as file:
            for line in file:
                legal_id.append(line.rstrip().lower())
        
        X_legal = []
        X_notlegal = []
        
        #from the company names, add name to X_legal, if it contains a legal_identifier
        #otherwise add it to X_notlegal, remove actual identifiers from string
        with open("companies.txt", 'r', encoding="utf8") as file:
            for line in file:
                sentence = " " + re.sub(r'\W+', ' ', line.rstrip().lower()) + " "
                is_legal = False
                for legal in legal_id:
                    if " " + legal + " " in sentence:
                        is_legal = True
                        idx = sentence.index(legal)
                        sentence = sentence[0: idx:] + sentence[idx +
                                        len(legal) + 1::]
                if is_legal == True:
                    X_legal.append(sentence)
                else:
                    X_notlegal.append(sentence)
        
        #Iterate over the labelled "legal" names and, for each word
        #in the entire training set, count how many of the comp_names contain the word
        vocab_words_notlegal = []
        distinct_words_as_sentences_notlegal = []
        
        #non-key words are not included
        irrelevant_words = ['the', 'of','your', 'a', 'de', 'and', 'e', 's']
        
        for sentence in X_notlegal:
            sentence_as_list = sentence.split()
            senten = []
            for word in sentence_as_list:
                if word not in irrelevant_words: #remove non-key words
                    vocab_words_notlegal.append(word) 
                    senten.append(word)
            distinct_words_as_sentences_notlegal.append(senten)
                
        vocab_words_legal = []
        distinct_words_as_sentences_legal = []
        
        for sentence in X_legal:
            sentence_as_list = sentence.split()
            senten = []
            for word in sentence_as_list:
                if word not in irrelevant_words:
                    vocab_words_legal.append(word)
                    senten.append(word)
            distinct_words_as_sentences_legal.append(senten)
                  
        total_words = vocab_words_notlegal + vocab_words_legal
        
        #total numbers of words in vocabulary
        total_word_num = len(list(dict.fromkeys(total_words)))
        
        #get unique words in both datasets
        vocab_unique_words_notlegal = list(dict.fromkeys(vocab_words_notlegal))
        vocab_unique_words_legal = list(dict.fromkeys(vocab_words_legal))
        
        #calculating spamicity of words in non-legal
        dict_nonlegal = {}
        total_notlegal = len(X_notlegal)
        
        dict_nonlegal = {}
        total_notlegal = len(X_notlegal)
        for sentence in X_notlegal:
            for word in sentence.split():
                if word not in irrelevant_words:
                    if word in dict_nonlegal:
                        dict_nonlegal[word] += 1
                    else:
                        dict_nonlegal[word] = 1
        
        #calculating spamicity of words in legal
        dict_legal = {}
        total_legal = len(X_legal)
        for sentence in X_legal:
            for word in sentence.split():
                if word not in irrelevant_words:
                    if word in dict_legal:
                        dict_legal[word] += 1
                    else:
                        dict_legal[word] = 1
        
        #Compute Probability of Notlegal and legal
        prob_notlegal = len(X_notlegal) / (len(X_notlegal)+(len(X_legal)))
        
        prob_legal = len(X_legal) / (len(X_notlegal)+(len(X_legal)))
        
        #Making a Classification
        def multi(probs) :        # function to multiply all word probs together 
            total_prob = 1
            for i in probs: 
                 total_prob = total_prob * i  
            return total_prob
        
        def Bayes(comp_name):
            probs = []
            probs2 = []
            for word in comp_name:
                try:
                    pr_WNL = dict_nonlegal[word]
                except KeyError:
                    # Apply smoothing for word not seen in NL training data, but seen in L training 
                    pr_WNL = 1/(total_notlegal+2)  
                    
                try:
                    pr_WL = dict_legal[word]
                except KeyError:
                    # Apply smoothing for word not seen in L training data, but seen in NL training
                    pr_WL = (1/(total_legal+2))  
                
                #prob_word_is_notlegal_BAYES = (pr_WNL*Pr_NL)/((pr_WNL*Pr_NL)+(pr_WL*Pr_L))
                prob_word_is_notlegal_BAYES = (pr_WNL + 1)/(len(vocab_unique_words_notlegal)
                                                            + total_word_num)
                
                prob_word_is_legal_BAYES = (pr_WL + 1)/(len(vocab_unique_words_legal)
                                                            + total_word_num)
                probs.append(prob_word_is_notlegal_BAYES)
                probs2.append(prob_word_is_legal_BAYES)
            
            #use Bayes theorem to get probabilities
            prob_notlegal_final = multi(probs) * prob_notlegal
            prob_legal_final = multi(probs2) * prob_legal
            
            #if probability of not containing a legal identifier is larger than that of
            #containing, return False, else return True
            if prob_notlegal_final > prob_legal_final:
                return False
            else:
                return True
        
        return Bayes(comp_name)
            

pred = NameComponents()
print(pred.get_name_components("Cognism (Germany) Ltd."))
print(pred.predict_legal_identifiers("Albemarle County Public Schools"))










