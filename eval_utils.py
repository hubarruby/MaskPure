from textattack.models.wrappers import ModelWrapper
from transformers import pipeline
import sys
sys.path.append('../')
from utils import *
sys.path.pop()

class MaskDemaskWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, fill_mask, num_voter, mask_pct, v_type):
        self.model = model
        self.tokenizer = tokenizer
        self.fill_mask = fill_mask
        self.num_voter = num_voter
        self.mask_pct = mask_pct
        self.pipeline = pipeline('text-classification', model=model, 
                                 tokenizer=tokenizer, device=next(model.parameters()).device)
        self.v_type = v_type
        
    def __call__(self, text_input_list):
        filled = mask_and_demask(text_input_list, self.tokenizer, self.fill_mask, verbose = False, 
                                 num_voter=self.num_voter, mask_pct=self.mask_pct
                                )
        # get the logits (to inform the attacker)
        logits = get_avg_logits(filled, self.pipeline, num_voter=self.num_voter, v_type=self.v_type)
        outputs = [[value for value in entry.values()] for entry in logits]
        return outputs
    
    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
    

def parse_attack_name(attack_name):
    """
    Function to parse attack name from the given attack object.

    Parameters:
    attack_name (object): Attack object

    Returns:
    string: Attack name as string
    """
    return f'{attack_name}'.split('.')[-1].strip("'>")


def convert_to_tuples(data):
    """
    Input data is of type datasets.Dataset
    For example, if you printed the first few of the input, it might look like:
    print(data[:2])
    {'text': ["I enjoyed the movie a lot!", 'Asolutely horrible film'], 'label': [1, 0]}
    
    Returns:
    Dataset in a form that textattack.datasets.Dataset can handle
    """
    tuples = []
    for text, label in zip(data['text'], data['label']):
        tuples.append((text, label))
    return tuples