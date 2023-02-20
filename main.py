import nltk as nltk
from happytransformer import HappyTextToText, TTSettings

from transformers import pipeline
from difflib import SequenceMatcher
from nltk.corpus import brown
import nltk

from nltk.tokenize import TreebankWordTokenizer as twt

# model = PunctuationModel()
happy_tt = HappyTextToText("T5",  "prithivida/grammar_error_correcter_v1")

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

import re
import os

from cleantext import clean

from tqdm.auto import tqdm
from transformers import pipeline


checker_model_name = "textattack/roberta-base-CoLA"
corrector_model_name = "pszemraj/flan-t5-large-grammar-synthesis"

# pipelines
checker = pipeline(
    "text-classification",
    checker_model_name,
)

# if os.environ.get("HF_DEMO_NO_USE_ONNX") is None:
#     # load onnx runtime unless HF_DEMO_NO_USE_ONNX is set
#     from optimum.pipelines import pipeline
#
#     corrector = pipeline("text2text-generation", model=corrector_model_name, accelerator="ort")
# else:
#     corrector = pipeline("text2text-generation", corrector_model_name)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
# tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")
# model.to(device)
# model.eval()

# def generate_text(text):
#     text = f'grammar: {text}'
#     input_ids = tokenizer(
#         text, return_tensors="pt"
#     ).input_ids
#     input_ids = input_ids.to(device)
#
#     outputs = model.generate(input_ids, max_length=512, early_stopping=True)
#
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

def puctutationRestroration(text):

    result = model.restore_punctuation(text)
    # print(result)
    return result


def splitByFullStop(text):
    # Use a breakpoint in the code line below to debug your script.
    x = text.split(".")
    # print(x)
    return x



# Press Ctrl+F8 to toggle the breakpoint.

def grammerCorrection(texts):
    corrected = []


    args = TTSettings(num_beams=5, min_length=1, max_length=500)

    # print(texts)
    # print('text............................')
    # Add the prefix "grammar: " before each input
    for text in texts:
        if text != '' and text != ' ':
            # text = correct_sentence_spelling(text)
            result = happy_tt.generate_text("grammar:" + text, args=args)
            # result = generate_text(text)
            print(result.text)
            corrected.append(result.text+ ' ')
            print(text)

    return corrected
     # This sentence has bad grammar.



def joinSentences(texts):
    textJoined = ''.join(texts)
    return textJoined

def summarization(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print(summarizer(text, max_length=130, min_length=30, do_sample=False))

# Press the green button in the gutter to run the script.
def tensIdentifier(texts):
    tensList = []
    for text in texts:
        text = text.lower()
        tokenized = nltk.word_tokenize(text)
        pronouns = [pos for (word, pos) in nltk.pos_tag(tokenized)]
        # pronouns =  nltk.pos_tag(tokenized)
        pronounstring = ''.join(pronouns)
        partsOfSpeechList = []
        if "VBZVBNVBG" in pronounstring:
            tensList.append('Present Perfect Continuous Tense')
        elif "VBDVBNVBG" in pronounstring:
            tensList.append('Past Perfect Continuous Tense')
        elif "MDVBVBNVBG" in pronounstring:
            tensList.append('Future Perfect Continuous Tense')


        elif "VBPVBN" in pronounstring:
            tensList.append('Present Continuous Tense')
        elif "VBDVBN" in pronounstring:
            tensList.append('Past Continuous Tense')
        elif "MDVBVBN" in pronounstring:
            tensList.append('Future Continuous Tense')


        elif "VBPVBG" in pronounstring:
            tensList.append('Present Perfect Tense')
        elif "VBDVBG" in pronounstring:
            tensList.append('Past Perfect Tense')
        elif "MDVBVBG" in pronounstring:
            tensList.append('Future Perfect Tense')

        elif "VBP" or "VBZ" in pronounstring:
            tensList.append('Present Tense')
        elif "VBD" in pronounstring:
            tensList.append('Past Tense')
        elif "MDVB" in pronounstring:
            tensList.append('Future Tense')

        if "MD" in pronounstring:
            partsOfSpeechList.append('Modal')

    #     print(pronounstring)
    # print(tensList)
    tensescore = len(set(tensList))
    print(tensescore)

def prepositionsIdentifier(texts):
    phraseList = []
    for text in texts:
        text = text.lower()
        # Tokenize text and pos tag each token
        tokens = twt().tokenize(text)
        tags = nltk.pos_tag(tokens, tagset="universal")
        Universalpronouns = [pos for (word, pos) in tags]
        UniversalpronounString = "".join(Universalpronouns)
        if "VERBADP" in UniversalpronounString and 'go to' not in text and not 'going to' in text:
            phraseList.append('phrase')
    print(phraseList)
    phrasescore = len(phraseList)
    print(phrasescore)
def comparetext(correctedText , originalText):
    return SequenceMatcher(None, correctedText, originalText).ratio()


from textblob import TextBlob


def correct_sentence_spelling(sentence):
    sentence = TextBlob(sentence)

    result = sentence.correct()
    print(result)
    return result


def ieltsWritingevaluation(fileName):
    textFile = open(fileName, "r")
    text = textFile.read()
    # punctuated = puctutationRestroration(text)
    splitedText = splitByFullStop(text)
    correctedText = grammerCorrection(splitedText)

    joinnedText = joinSentences(correctedText)
    # sentenseText = ['I play a game every day.',' I am playing the game.',
    #                 'I have finished my homework.','He has been studying in the school since his childhood.',
    #                 'You played the game.','I was reading a newspaper. ',
    #                 ' I had finished my homework. ','I had been finishing my homework for 50 minutes.',
    #                 'I shall go to my home town.','I will be watching the news at 9 pm.',
    #                 'I will have played the game.','I will have been watching the news for over ten minutes before you join me.']
    # tensIdentifier(splitedText)
    prepositionsIdentifier(splitedText)
    # print('Original')
    # print(text.strip('\n'))
    # print('corrected')
    # print(joinnedText)
    matchingPercentage = comparetext(joinnedText, text.strip('\n'))
    print(fileName)
    print(matchingPercentage)


#correction with grammer synthesis

# Initialize the text-classification pipeline
# from transformers import pipeline
# checker = pipeline(
#         'text-classification',
#         'textattack/roberta-base-CoLA'
#     )

# Initialize the text-generation pipeline
# from transformers import pipeline
# corrector = pipeline(
#         "text2text-generation",
#         "pszemraj/flan-t5-large-grammar-synthesis",
#     )

# Test the function with a sample text
raw_text = "the toweris 324 met (1,063 ft) tall, about height as .An 81-storey building, and biggest longest structure paris. Is square, measuring 125 metres (410 ft) on each side. During its constructiothe eiffel tower surpassed the washington monument to become the tallest man-made structure in the world, a title it held for 41 yearsuntilthe chryslerbuilding in new york city was finished in 1930. It was the first structure to goat a height of 300 metres. Due 2 the addition ofa brdcasting aerial at the t0pp of the twr in 1957, it now taller than  chrysler building 5.2 metres (17 ft). Exxxcluding transmitters,  eiffel tower is  2ndd tallest ree-standing structure in france after millau viaduct."

import pprint as pp
pp.pprint(raw_text)
if __name__ == '__main__':
    # ieltsWritingevaluation("ieltsscore4.txt")
    # ieltsWritingevaluation("ieltsscore5.txt")
    # ieltsWritingevaluation("ieltsscore6.txt")
    ieltsWritingevaluation("ieltsscore7.txt")
    # ieltsWritingevaluation("ieltsscore8.txt")
    # # correct_sentence_spelling('He has done his home work');
    # grammerCorrection(['heloo dear','Free time in my opinion refers to time not spent under the direct supervision of '
    #                                 'a parent, teacher or a version enthrusted with the responsibility of bringing up'
    #                                 ' the child.'])
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
