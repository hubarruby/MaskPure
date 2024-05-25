import nltk
import numpy as np
import random
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import torch
import datasets
import pandas as pd
import random
from transformers import pipeline, BertTokenizer, BertForMaskedLM

random.seed(hash('Harrison Gietz'))

def mask_random_tokens(text, tokenizer, mask_pct = 0.2, n = 5, pos_weights=None, return_separated = False):
    """ 
    Masks a fraction of tokens in a given text string, generating multiple (n) masked versions.

    This function tokenizes the input string and randomly replaces a percentage of the tokens with a
    mask token in multiple copies of the text. The likelihood of masking each token can be adjusted 
    by specifying weights based on the part-of-speech (POS) tags of the tokens. The function 
    operates in a vectorized manner for improved performance.

    Parameters:
    text (str): The input string to mask.
    tokenizer (object): The tokenizer to use, typically "bert-base-uncased" for our purposes.
    mask_pct (float, optional): The percentage of tokens to mask in each copy. Defaults to 0.2.
    n (int, optional): The number of masked copies to create. Defaults to 5.
    pos_weights (dict, optional): A dictionary mapping POS tags to weights. If specified, tokens
        with the corresponding POS tags will be more likely to be masked in all copies. The weights 
        should be greater than or equal to 1. For example, {'NN': 2.0, 'JJ': 1.5} will make nouns 
        twice as likely and adjectives 50% more likely to be masked compared to other tokens. 
        Defaults to None, which results in unweighted random masking.
    return_separated (bool, optional): if True, outputs an additional array of tokenized texts along with the joined texts 

    Returns:
    np.ndarray: An n-by-1 numpy array of masked text strings.
    """
    nltk_tokens = np.array(word_tokenize(text), dtype = object)
    n_copies = np.tile(nltk_tokens, (n, 1))

    pos_tags = nltk.pos_tag(nltk_tokens)
    weights = [1.0 for _ in range(len(pos_tags))]
    if pos_weights is not None:
        for i, (_, pos) in enumerate(pos_tags):
            if pos in pos_weights:
                weights[i] = pos_weights[pos]
                
    weights = np.array(weights) / np.sum(weights)
    n_to_mask = max(2, round(len(nltk_tokens) * mask_pct))

    mask_indices = np.array([np.random.choice(range(len(nltk_tokens)), size=n_to_mask, p=weights, replace=False)
                             for i in range(n)])
    
    n_copies[np.arange(n)[:,None], mask_indices] = "[MASK]"
    v_convert_tokens_to_string = np.vectorize(tokenizer.convert_tokens_to_string, signature='(n)->()', otypes = [object])
    masked_text = v_convert_tokens_to_string(n_copies).reshape(-1,1)
    
    if return_separated:
        return masked_text, n_copies
    
    return masked_text

def find_words_in_double_brackets(input_data):
    if isinstance(input_data, str):
        # If input_data is a single string
        pattern = r'\[\[(.*?)\]\]'
        matches = re.findall(pattern, input_data)
        return matches
    elif isinstance(input_data, pd.Series):
        # If input_data is a pandas Series
        pattern = r'\[\[(.*?)\]\]'
        matches = input_data.str.findall(pattern)
        return matches
    else:
        raise ValueError("Input data must be a string or a pandas Series.")
        
def remove_double_brackets(input_data):  
    """ 
    Takes a string or pd.Series as input
    Returns the input with all '[[' and ']]' removed from the text
    """
    if isinstance(input_data, str):
        # If input_data is a single string
        return re.sub(r'\[\[|\]\]', '', input_data)
    elif isinstance(input_data, pd.Series):
        # If input_data is a pandas Series
        return input_data.str.replace(r'\[\[(.*?)\]\]', r'\1')
    else:
        raise ValueError("Input data must be a string or a pandas Series.")

def sort_words_by_pos(input_list):
    tagged_words = []
    for entry in input_list:
        words = nltk.word_tokenize(entry)
        word_tags = nltk.pos_tag(words)
        tagged_words.extend(word_tags)
    
    return tagged_words

def plot_most_common_pos_list(pos_list_dict, text_type, indv_plots = True, norm = False, take_top = True):
    """
    Plots the most common Part-Of-Speech (POS) tags in the given text.

    Parameters:
    pos_list_dict (dict): A dictionary where the keys are the names/ids of the texts and the values are lists of (word, POS) tuples.
    text_type (str, optional): Specifies which texts to include in the plot. Options are 'both', 'og', 'adv'. 
    indv_plots (bool, optional): If True, creates individual plots for each text. If False, combines all texts into one plot.
    Defaults to True.
    norm (bool, optional): If True, normalizes the POS counts to between 0 and 1 using MinMaxScaler. Defaults to False.
    take_top (bool, optional): If True, only plots the top 5 most common POS types. If False, plots all POS types. Defaults to True.

    Returns:
    None

    This function generates bar plots of the most common POS tags in the given text or texts. The number of occurrences of each POS tag is
    used as the y-value for the bar plot. 
    The text_type parameter determines which texts to include in the plot(s). The options are 'both', 'og', and 'adv', with 'both' being the
    default.
    If indv_plots is True, the function generates individual plots for each text. Otherwise, it combines the counts from all texts into one
    plot.
    If norm is True, the POS counts are normalized to between 0 and 1 using MinMaxScaler. Otherwise, the raw counts are used.
    If take_top is True, the function only plots the top 5 most common POS types. Otherwise, it plots all POS types.
    """

    if text_type == 'both':
        iter_keys = pos_list_dict.keys()
    elif text_type == 'og':
        iter_keys = sort_keys_by_string(pos_list_dict, 'og')
    elif text_type == 'adv':
        iter_keys = sort_keys_by_string(pos_list_dict, 'adv')
    # rest of your code
    
    if indv_plots:
        for key in iter_keys:
            pos_list = pos_list_dict.get(key)
            pos_counts = {}
            for word, pos in pos_list:
                if pos in pos_counts:
                    pos_counts[pos] += 1
                else:
                    pos_counts[pos] = 1

            sorted_pos_counts = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
            
            if take_top:
                top_pos = sorted_pos_counts[:5]  # Choose the top 5 most common POS types
            else:
                top_pos = sorted_pos_counts
                plt.figure(figsize = (20,6))

            pos_types = [pos[0] for pos in top_pos]
            counts = [pos[1] for pos in top_pos]
            
            if norm:
                # Create the MinMaxScaler object and fit_transform the counts
                scaler = MinMaxScaler(feature_range=(0, 1))
                counts = scaler.fit_transform([[count] for count in counts]).flatten()

            plt.bar(pos_types, counts)
            plt.xlabel('POS Types')
            plt.ylabel('Counts')
            plt.title(f'Most Common POS: {key}')
            plt.show()
    else:
        pos_counts = {}
        for key in iter_keys:
            pos_list = pos_list_dict.get(key)
            for word, pos in pos_list:
                if pos in pos_counts:
                    pos_counts[pos] += 1
                else:
                    pos_counts[pos] = 1

        sorted_pos_counts = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
        
        if take_top:
            top_pos = sorted_pos_counts[:5]  # Choose the top 5 most common POS types
        else:
            top_pos = sorted_pos_counts
            plt.figure(figsize = (20,6))
            
        pos_types = [pos[0] for pos in top_pos]
        counts = [pos[1] for pos in top_pos]
        
        if norm:
            # Create the MinMaxScaler object and fit_transform the counts
            scaler = MinMaxScaler(feature_range=(0, 1))
            counts = scaler.fit_transform([[count] for count in counts]).flatten()
        
        plt.bar(pos_types, counts)
        plt.xlabel('POS Types')
        plt.ylabel('Counts')
        plt.title(f'Most Common POS Types: {text_type} texts')
        plt.show()

def remove_special_characters(text):  
    """
    Removes special character sequences '[[' and ']]' from the input string.
    
    Parameters:
    text (str): The input string.

    Returns:
    str: The modified string with special characters removed.
    """
    text = re.sub(r'\[\[|\]\]', '', text)
    return text

def csv_processing(csv_data): 
    """
    Processes CSV data generated by Textattack, removing [[ ]] characters that were added by the Textattack attacks surrounding adversarial pertubations.
    
    Parameters:
    csv_data (DataFrame): The input DataFrame generated by the log_csv function from the Textattack library.

    Returns:
    csv_data (Dataset): hugginface dataset
    """
    total_attacks = 0
    successes = 0
#   count attack success rate and get adversarial data available for defensive cleaning afterward
    csv_data['model_correct_perturbed'] = (csv_data['ground_truth_output'] == csv_data['perturbed_output']).astype(int)
    csv_data['model_correct_original'] = (csv_data['ground_truth_output'] == csv_data['original_output']).astype(int)
    csv_data.rename(columns={"ground_truth_output": "label", 'perturbed_text': 'text'}, inplace = True)
    csv_data.drop(columns = ['original_score', 'perturbed_score', 'num_queries', 'result_type'], inplace= True)
    csv_data['text'] = csv_data['text'].map(remove_special_characters)
    csv_data['original_text'] = csv_data['original_text'].map(remove_special_characters)
    csv_data = datasets.Dataset.from_pandas(csv_data)

    return csv_data

def filter_and_eval_csv_data(csv_data, tokenizer, verbose=False, max_length=435):
    """
    This is en explanation of the function
    """
    
    def filter_length(example):
        """
        Checks if the length of the tokenized 'text' from the input dictionary is within the specified maximum length.

        Parameters:
        example (dict): A dictionary containing the key 'text'.

        Returns:
        bool: True if the number of tokens in 'text' is less than or equal to max_length, otherwise False.
        """
        return len(tokenizer.tokenize(example['text'])) <= max_length
    
    csv_data = csv_data.filter(filter_length)
    
    attack_acc_original = float(torch.tensor(csv_data['model_correct_original']).sum()/len(csv_data)*100)
    attack_acc_perturbed = float(torch.tensor(csv_data['model_correct_perturbed']).sum()/len(csv_data)*100)
    
    if verbose:
        print(f'model accuracy before attacks: {round(int(attack_acc_original), 4)}%')
        print(f'model accuracy after attacks: {round(int(attack_acc_perturbed), 4)}%')
    return attack_acc_original, attack_acc_perturbed, csv_data

    
def find_max_labels(dictionary):
    """
    Finds all keys with the maximum value in the given dictionary.
    Used for finding the max label(s) during the logit voting method.

    Args:
        dictionary (dict): A dictionary containing key-value pairs.

    Returns:
        list: A list of keys with the maximum value. If multiple keys have
              the same maximum value, all of them will be included in the list.
    """
    max_value = float('0')
    max_keys = []

    for key, value in dictionary.items():
        if value > max_value:
            max_value = value
            max_keys = [key]
        elif value == max_value:
            max_keys.append(key)

    return max_keys

def voting(inputs, pipeline, num_voter = 5, v_type='majority'):
    """
    Implements a voting strategy on a list of text inputs.
    
    Parameters:
    inputs (list): list of input strings for a classifier pipeline. len = (n*num_voter)
    pipeline (Tranformers pipeline): a pipeline for sequence classification.
    num_voter (int, optional): The number of entries to consider in each vote. Defaults to 5.
    v_type (str, optional): One of "majority", "logit", "maj_log". Determines whether majority voting, logit averaging voting, or 
            "majority voting with logits for breaking ties" will be conducted (default is majority)

    Returns:
    list: A list of string results from the voting strategy.
    """
    if v_type == 'majority':
        if pipeline.model.config.num_labels != 2:
            raise ValueError('Cannot implement pure majority voting for a pipeline with num_label != 2. '
                 'Try v_type = "logit" or "maj_log"')
        results = pipeline(inputs)
        final_results = []
        for i in range(0, len(results), num_voter):
            positive_votes = sum(1 for result in results[i:i+num_voter] if result['label'] in ['LABEL_1', '1', 'POSITIVE', 1])
            if positive_votes > num_voter//2:
                final_results.append('1')
            else:
                final_results.append('0')
        return final_results
    
    elif v_type == 'logit':
        results = pipeline(inputs, top_k = None) 
        final_results = []

        if len(results[0]) != pipeline.model.config.num_labels:
            raise ValueError(f'Pipeline number of labels ({pipeline.model.config.num_labels}) '
                             f'and inner input text list length ({len(results)} outer and {len(results[0])} inner) '
                             ' must have matching dims')
        else: 
            num_labels = len(results[0])
        
        # iterate over input list with stepsize n
        for i in range(0, len(results), num_voter):
            sublist = results[i: i + num_voter]
            if num_labels == 2:
                avg_scores = {'LABEL_0': [], 'LABEL_1': []}
            elif num_labels == 4:
                avg_scores = {'LABEL_0': [], 'LABEL_1': [], 'LABEL_2': [], 'LABEL_3': []}
            else: 
                raise ValueError(f'Unsupported number of labels ({num_labels}) in your pipeline for logit voting. '
                                 'Requires 2 (imdb) or 4 (agnews)')
            for dict_list in sublist:
                for dic in dict_list:
                    avg_scores[dic['label']].append(dic['score'])
            for label in avg_scores.keys():
                avg_scores[label] = np.mean(avg_scores[label])
            final_results.append(find_max_labels(avg_scores)[0])
                
        return final_results

    elif v_type== 'maj_log':
        results = pipeline(inputs,top_k=None)
        final_results = []
        
        if len(results[0]) != pipeline.model.config.num_labels:
            raise ValueError(f'Pipeline number of labels ({pipeline.model.config.num_labels}) '
                             f'and inner input text list length ({len(results)} outer and {len(results[0])} inner) '
                             ' must have matching dims')
        else: 
            num_labels = len(results[0])

        #Try majority voting (no logits involved) first
        for i in range(0, len(results), num_voter):
            sublist = results[i: i + num_voter]
            if num_labels == 2:
                vote_tally = {'LABEL_0': 0, 'LABEL_1': 0}
            elif num_labels == 4:
                vote_tally = {'LABEL_0': 0, 'LABEL_1': 0, 'LABEL_2': 0, 'LABEL_3': 0}
            else: 
                raise ValueError(f'Unsupported number of labels ({num_labels}) in your pipeline for logit voting. '
                                 'Requires 2 (imdb) or 4 (agnews)')

            for dict_list in sublist:
                top_score_val = 0
                top_score_label = None
                for dic in dict_list:
                    if dic['score'] > top_score_val:
                        top_score_val = dic['score']
                        top_score_label = dic['label']
                vote_tally[top_score_label] += 1
#             print('vote_tally', vote_tally)
            top_label_overall = find_max_labels(vote_tally)
#             print('top label', top_label_overall)
#             print()
            if len(top_label_overall) > 1:
                #implement logit voting, since two keys tied for majority voting:
                if num_labels == 2:
                    avg_scores = {'LABEL_0': [], 'LABEL_1': []}
                elif num_labels == 4:
                    avg_scores = {'LABEL_0': [], 'LABEL_1': [], 'LABEL_2': [], 'LABEL_3': []}
                else: 
                    raise ValueError(f'Unsupported number of labels ({num_labels}) in your pipeline for logit voting '
                                     '(inside of maj_log). '
                                     'Requires 2 (imdb) or 4 (agnews)')
                for dict_list in sublist:
                    for dic in dict_list:
                        avg_scores[dic['label']].append(dic['score'])
                for label in avg_scores.keys():
                    avg_scores[label] = np.mean(avg_scores[label])
                final_results.append(find_max_labels(avg_scores)[0])
                
            else:
                final_results.append(top_label_overall[0])
            
        return final_results
    
    else:
        raise ValueError(f'Parameter v_type must be one of "majority", "logit", or "maj_log".')
        

def mask_and_demask(filtered_dataset_text, 
                    tokenizer,
                    fill_mask,
                    num_voter=5, 
                    verbose=True,
                    mask_pct=0.2,
                    pos_weights=None):
    """
    Applies a process of masking and demasking on input data, repeating the process for a specified number of times for each sample.
    
    Parameters:
    filtered_dataset_text (list): List of text strings from the filtered dataset.
    tokenizer (object): The tokenizer to use.
    num_voter (int, optional): Number of times the mask and demask process is repeated per sample. Defaults to 5.
    verbose (bool, optional): Whether to print processing steps. Defaults to True.
    fill_mask (transformers.pipeline, optional): The fill-mask pipeline for unmasking tokens. Defaults to pipeline("fill-mask", model="bert-base-uncased", tokenizer="bert-base-uncased").
    mask_pct (float, optional): The percentage of tokens to mask. Defaults to 0.2.
    pos_weights (dict, optional): A dictionary mapping POS tags to weights. If specified, tokens with the corresponding POS tags will be more likely to be masked.

    Returns:
    np.ndarray: An array of modified text strings.
    """
    modified_adv_texts = []
    v_convert_tokens_to_string = np.vectorize(tokenizer.convert_tokens_to_string, signature='(n)->()', otypes = [object])
    for i, example in enumerate(filtered_dataset_text):
        if i % 25 == 0 and verbose:
            print(f'Done processing {i}/{len(filtered_dataset_text)} adversarial samples.')
        # Generate all masked versions in one operation, for each text
        # unfortunately, this cannot be parallelized for multiple strings because nltk (used inside the function) runs on CPU only
        # ...and is not compatible with np arrays
        masked_texts, tokenized_masked_texts = mask_random_tokens(example, 
                                                                tokenizer, 
                                                                mask_pct=mask_pct, 
                                                                n=num_voter, 
                                                                pos_weights=pos_weights, 
                                                                return_separated=True)
        replace_idxs = np.argwhere(tokenized_masked_texts == '[MASK]')
        # Unmask the texts and save the results
        unmasked_text_suggestions = fill_mask([list(masked_text) for masked_text in masked_texts], top_k = 1)
        replacement_tokens = [token_info[0]['token_str']  
                              for sentence in unmasked_text_suggestions for token_info in sentence]
        tokenized_masked_texts[replace_idxs[:, 0], replace_idxs[:, 1]] = replacement_tokens
        unmasked = v_convert_tokens_to_string(tokenized_masked_texts).reshape(-1,)
        [modified_adv_texts.append(unmasked[i]) for i in range(num_voter)]
        
    return modified_adv_texts


def get_avg_logits(inputs, pipeline, num_voter, v_type='logit', softmax_after_tally=False):
    """
    Similar to voting(), but uses only logits and returns all logit values, not just the top label.
    Used for creating input logits into textattack for generating adv attacks.
    
    v_type (optional): 'logit', 'majority', or 'maj_log'. Determines how the final logits are presented.
        if 'logit': 
            decisions are made based on averaged logits, and the averaged logits are what will be outputted 
        
        if 'majority': 
            (can only be used for imdb/2-class problems) Decisions are made based on majority voting, 
            and the output logits are based on voting fraction 
            (i.e. if there are 4 votes for positive and 1 vote for negative, that means the "logits" outputed would be [0.8, 0.2])
            
        if 'maj_log':
            (similar to majority, but adapted to work for ag_news/multi-class problems)
            Decisions made based on majority voting by default, but if there is a tie, then it's broken with averaged logits
            (i.e. if the votes are {label_0: 2, label_1: 4, label_2: 0, label_3: 1} the output "logits" would be [2/7, 4/7, 0, 1/7].
            If the votes are {label_0: 3, label_1: 3, label_2: 0, label_3: 1}, the output would be the averaged logits across each class.
            
        if 'maj_one_hot':
            basicially a version of maj_log where there's no aproximation
            (i.e. if the majority vote is for class 1, then the outputs is {label_0: 0, label_1: 1} reardless of exact vote numbers
            
    softmax_afer_tally (optional): Only compatible with v_type=majority or v_type=maj_log. To emulate logits more authentically,
        instead of returning relative voting scores as logits, puts final vote scores through a softmax function i.e. (0.2, 0.8) 
        before outputting. 
        Example: 
            5 voters. 2 classes. label_0 has 1 vote; label_1 has 4 votes.
            if softmax_after_tally = False, the output is [1/5, 4/5] = [0.2, 0.8]
            if softmax_after_tally = True, the output is softmax([1, 4]) = [0.047..., 0.95...]
    """
    if v_type =='logit' and softmax_after_tally == True:
        raise ValueError('v_type="logit" and softmax_after_tally = True are not compatible; must use v_type = "majority" or "maj_log"'
                         ', else turn softmax_after_tally=False. ')
    
    if v_type == 'logit':
        results = pipeline(inputs, top_k = None) 
        final_results = []

        if len(results[0]) != pipeline.model.config.num_labels:
            raise ValueError(f'Pipeline number of labels ({pipeline.model.config.num_labels}) '
                             f'and inner input text list length ({len(results)} outer and {len(results[0])} inner) '
                             ' must have matching dims')
        else: 
            num_labels = len(results[0])

        # iterate over input list with stepsize n
        for i in range(0, len(results), num_voter):
            sublist = results[i: i + num_voter]
            if num_labels == 2:
                avg_scores = {'LABEL_0': [], 'LABEL_1': []}
            elif num_labels == 4:
                avg_scores = {'LABEL_0': [], 'LABEL_1': [], 'LABEL_2': [], 'LABEL_3': []}
            else: 
                raise ValueError(f'Unsupported number of labels ({num_labels}) in your pipeline for averaging logits. '
                                 'Requires 2 (imdb) or 4 (agnews)')
            for dict_list in sublist:
                for dic in dict_list:
                    avg_scores[dic['label']].append(dic['score'])
            for label in avg_scores.keys():
                avg_scores[label] = np.mean(avg_scores[label])
            final_results.append(avg_scores)

        return final_results
    
    elif v_type == 'majority':
        if pipeline.model.config.num_labels != 2:
            raise ValueError('Cannot implement pure majority voting for a pipeline with num_label != 2. '
                 'Try v_type = "logit" or "maj_log"')
        results = pipeline(inputs)
        final_results = []
        for i in range(0, len(results), num_voter):
            positive_votes = sum(1 for result in results[i:i+num_voter] if result['label'] in ['LABEL_1', '1', 'POSITIVE', 1])
            #output logits based on voter ratio (instead of raw averaged logits)
            if not softmax_after_tally:
                final_results.append({'LABEL_0': (num_voter - positive_votes)/num_voter, 'LABEL_1': positive_votes/num_voter})
            else:
                outputs = torch.nn.functional.softmax(torch.tensor([float(num_voter - positive_votes), 
                                                                    float(positive_votes)]), dim=0).tolist()
                final_results.append({'LABEL_0': outputs[0], 'LABEL_1': outputs[1]})
        return final_results
    
    elif v_type== 'maj_log':
        results = pipeline(inputs,top_k=None)
        final_results = []
        
        if len(results[0]) != pipeline.model.config.num_labels:
            raise ValueError(f'Pipeline number of labels ({pipeline.model.config.num_labels}) '
                             f'and inner input text list length ({len(results)} outer and {len(results[0])} inner) '
                             ' must have matching dims')
        else: 
            num_labels = len(results[0])

        #Try majority voting (no logits involved) first
        for i in range(0, len(results), num_voter):
            sublist = results[i: i + num_voter]
            if num_labels == 2:
                vote_tally = {'LABEL_0': 0, 'LABEL_1': 0}
            elif num_labels == 4:
                vote_tally = {'LABEL_0': 0, 'LABEL_1': 0, 'LABEL_2': 0, 'LABEL_3': 0}
            else: 
                raise ValueError(f'Unsupported number of labels ({num_labels}) in your pipeline for logit voting. '
                                 'Requires 2 (imdb) or 4 (agnews)')

            for dict_list in sublist:
                top_score_val = 0
                top_score_label = None
                for dic in dict_list:
                    if dic['score'] > top_score_val:
                        top_score_val = dic['score']
                        top_score_label = dic['label']
                vote_tally[top_score_label] += 1
#             print('vote_tally', vote_tally)
            top_label_overall = find_max_labels(vote_tally)

            if len(top_label_overall) > 1:
                # Implement logit voting, since there was a tie by discrete votes
                if num_labels == 2:
                    avg_scores = {'LABEL_0': [], 'LABEL_1': []}
                elif num_labels == 4:
                    avg_scores = {'LABEL_0': [], 'LABEL_1': [], 'LABEL_2': [], 'LABEL_3': []}
                else: 
                    raise ValueError(f'Unsupported number of labels ({num_labels}) in your pipeline for averaging logits. '
                                     'Requires 2 (imdb) or 4 (agnews)')
                for dict_list in sublist:
                    for dic in dict_list:
                        avg_scores[dic['label']].append(dic['score'])
                for label in avg_scores.keys():
                    avg_scores[label] = np.mean(avg_scores[label])
                final_results.append(avg_scores)
                
            else:
                if not softmax_after_tally:
                    # Calculate the total sum
                    total = sum(vote_tally.values())
                    # Create a new dictionary where the values are divided by the total
                    normalized_vote_tally = {k: v/total for k, v in vote_tally.items()}
                    final_results.append(normalized_vote_tally)
                else:
                    vote_tally_tensor = torch.tensor([float(vote_tally[key]) for key in vote_tally.keys()])
#                     print('vote tally tensor: ', vote_tally_tensor)
                    outputs = torch.nn.functional.softmax(vote_tally_tensor, dim=0).tolist()
                    final_results.append({k: outputs[i] for i, k in enumerate(vote_tally.keys())})
            
        return final_results
    
    elif v_type== 'maj_one_hot':
        results = pipeline(inputs,top_k=None)
        final_results = []
        
        if len(results[0]) != pipeline.model.config.num_labels:
            raise ValueError(f'Pipeline number of labels ({pipeline.model.config.num_labels}) '
                             f'and inner input text list length ({len(results)} outer and {len(results[0])} inner) '
                             ' must have matching dims')
        else: 
            num_labels = len(results[0])

        #Try majority voting (no logits involved) first
        for i in range(0, len(results), num_voter):
            sublist = results[i: i + num_voter]
            if num_labels == 2:
                vote_tally = {'LABEL_0': 0, 'LABEL_1': 0}
            elif num_labels == 4:
                vote_tally = {'LABEL_0': 0, 'LABEL_1': 0, 'LABEL_2': 0, 'LABEL_3': 0}
            else: 
                raise ValueError(f'Unsupported number of labels ({num_labels}) in your pipeline for logit voting. '
                                 'Requires 2 (imdb) or 4 (agnews)')

            for dict_list in sublist:
                top_score_val = 0
                top_score_label = None
                for dic in dict_list:
                    if dic['score'] > top_score_val:
                        top_score_val = dic['score']
                        top_score_label = dic['label']
                vote_tally[top_score_label] += 1
#             print('vote_tally', vote_tally)
            top_label_overall = find_max_labels(vote_tally)
#             print(top_label_overall)
            if len(top_label_overall) > 1:
                # Implement logit voting, since there was a tie by discrete votes
                if num_labels == 2:
                    avg_scores = {'LABEL_0': [], 'LABEL_1': []}
                elif num_labels == 4:
                    avg_scores = {'LABEL_0': [], 'LABEL_1': [], 'LABEL_2': [], 'LABEL_3': []}
                else: 
                    raise ValueError(f'Unsupported number of labels ({num_labels}) in your pipeline for averaging logits. '
                                     'Requires 2 (imdb) or 4 (agnews)')
                for dict_list in sublist:
                    for dic in dict_list:
                        avg_scores[dic['label']].append(dic['score'])
                for label in avg_scores.keys():
                    avg_scores[label] = np.mean(avg_scores[label])
                final_results.append(avg_scores)
                
            else:
                contrived_logits = {k: 0 for k in vote_tally.keys()}
                contrived_logits[top_label_overall[0]] = 1
                final_results.append(contrived_logits)
#                 print(contrived_logits)
        return final_results
    
    else:
        raise ValueError(f'Parameter v_type must be one of "majority", "logit", or "maj_log".')

def eval_masking(filtered_dataset, 
                 classifier_pipeline,
                 tokenizer,
               mask_pct = 0.2, 
               num_voter = 5, 
               fill_mask = pipeline("fill-mask", model="bert-base-uncased", tokenizer="bert-base-uncased"),
               pos_weights = None,
               verbose = True,
               eval_og_data = True):
    """
    Evaluates the performance of a defense mechanism against adversarial attacks on a given dataset.

    Parameters:
    filtered_dataset (datasets.dataset): A dataset (with all texts below 512 tokens) to be processed and evaluated.
    classifier_pipeline (transformers.pipeline): the model and tokenizer pipeline to classify final results.
    tokenizer: 
    mask_pct (float, optional): The percentage of tokens to be masked in the text data. Default is 0.2.
    num_voter (int, optional): The number of times the mask and demask process is repeated for each sample. Default is 5.
    fill_mask (transformers.pipeline, optional): A transformers pipeline object used for masking and demasking. Default is "fill-mask" pipeline with "bert-base-uncased" model and tokenizer.
    verbose (bool, optional): Controls the print statements for progress and performance. If True, print statements are displayed. Default is True.
    eval_og_data (bool, optional): If True, the function also evaluates the performance of the defense mechanism on the non-adversarial (original) data. Default is True.

    Returns:
    final_accs (dict): A dictionary containing the accuracies of the model after defense against adversarial and, optionally, non-adversarial data. The dictionary contains the keys 'adv_after_defense' and 'clean_after_defense'.

    The function applies a defense mechanism of random masking, filling, and majority voting against adversarial attacks. It evaluates the performance of this mechanism and returns the final accuracies. It proceeds to mask and demask a percentage of the tokens in the text data, repeating this process num_voter times for each sample. If eval_og_data is set to True, the function also evaluates the impact of the defense mechanism on non-adversarial data.
    """
    final_accs = {'adv_after_defense': None, 'clean_after_defense': None}
#     take the dataset and mask + demask random tokens num_voter times per sample
#  input dataset should just be a single list/column
    modified_adv_texts = mask_and_demask(filtered_dataset['text'], 
                                        tokenizer,
                                        num_voter = num_voter, 
                                        verbose = verbose, 
                                        fill_mask = fill_mask,
                                        mask_pct = mask_pct,
                                        pos_weights = pos_weights)
    # Perform sentiment analysis
    adv_unvoted_results = classifier_pipeline(modified_adv_texts)
    if verbose: print('Getting accuracy results from voting...')
    final_adv_results = majority_voting(adv_unvoted_results, num_voter = num_voter)
    # comparing predicitions to true values; getting accuracy for adversarial text, once defended
    adv_correct_pred = 0
    for i, result in enumerate(final_adv_results):
        true = filtered_dataset["label"][i]
        # print(f'Review {i+1}: {result}. True label: {true}')
        if int(result) == int(true):
            adv_correct_pred += 1
    # Evaluate accuracy
    final_accs['adv_after_defense'] = adv_correct_pred / len(filtered_dataset['label']) * 100
    
    # evaluate the diff in accuracy for the clean data; see how negatively the defense affects normal performance
    if eval_og_data:
        if verbose: print('Evaluating defense method on unperturbed data, to see how it affects performance...')
        modified_clean_texts = mask_and_demask(filtered_dataset['original_text'],
                                                tokenizer,
                                                num_voter = num_voter, 
                                                verbose = verbose, 
                                                fill_mask = fill_mask,
                                                mask_pct = mask_pct,
                                                pos_weights = pos_weights)
        # Perform sentiment analysis
        clean_unvoted_results = classifier_pipeline(modified_clean_texts)
        if verbose: print('Getting accuracy results from voting...')
        final_clean_results = majority_voting(clean_unvoted_results, num_voter = num_voter)
        # copmaring predicitions to true values; getting accuracy for adversarial text, once defended
        clean_correct_pred = 0
        for i, result in enumerate(final_clean_results):
            true = filtered_dataset["label"][i]
            # print(f'Review {i+1}: {result}. True label: {true}')
            if int(result) == int(true):
                clean_correct_pred += 1
        # Evaluate accuracy
        final_accs['clean_after_defense'] = clean_correct_pred / len(filtered_dataset['label']) * 100
        
    return final_accs