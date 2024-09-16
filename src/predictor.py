# %%
import os
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
# from cv2 import dnn_superres
import re
import joblib
import re
import pandas as pd
from tqdm import tqdm
from utils import download_image
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings




# %%
AREA_THRESHOLD = 500000
# Suppress the RNN warning
warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk of memory")
reader = easyocr.Reader(['en'])
tqdm.pandas()

# %%
loaded_model = joblib.load('best_model.pkl')
label_encoder_group = joblib.load('label_encoder_group.pkl')
label_encoder_entity = joblib.load('label_encoder_entity.pkl')

# %%
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

# %%
abbreviation_map = {
    'cm': 'centimetre', 'CM': 'centimetre',
    'ft': 'foot', 'FT': 'foot',
    'in': 'inch', 'IN': 'inch', 'inch.': 'inch',
    'm': 'metre', 'M': 'metre',
    'mm': 'millimetre',
    'yd': 'yard',
    'g': 'gram', 'G': 'gram',
    'KG': 'kilogram', 'kg': 'kilogram', 'Kg': 'kilogram',
    'mcg': 'microgram', 'MCG': 'microgram',
    'mg': 'milligram', 'MG': 'milligram',
    'oz': 'ounce', 'OZ': 'ounce',
    'lbs': 'pound', 'LBS': 'pound',
    'ton': 'ton', 'TON': 'ton',
    'kV': 'kilovolt', 'kv': 'kilovolt', 'KV': 'kilovolt',
    'mV': 'millivolt', 'mv': 'millivolt', 'MV': 'millivolt',
    'v': 'volt', 'V': 'volt',
    'kw': 'kilowatt', 'kW': 'kilowatt', 'Kw': 'kilowatt',
    'w': 'watt', 'W': 'watt',
    'cl': 'centilitre', 'Cl': 'centilitre',
    'cu ft': 'cubic foot',
    'cups': 'cup', 'CUPS': 'cup',
    'dl': 'decilitre', 'dL': 'decilitre', 'Dl': 'decilitre',
    'fl oz': 'fluid ounce',
    'gal': 'gallon',
    'imp gal': 'imperial gallon',
    'l': 'litre', 'L': 'litre',
    'ul': 'microlitre', 'uL': 'microlitre',
    'ml': 'millilitre', 'mL': 'millilitre', 'Ml': 'millilitre',
    'pt': 'pint',
    'qt': 'quart'
}

# %%
def preprocess(image):
    def calArea(img, area):
        if area < AREA_THRESHOLD:
            scale_percent = 300  # percent of original size
            new_width = int(img.shape[1] * scale_percent / 100)
            new_height = int(img.shape[0] * scale_percent / 100)
            dim = (new_width, new_height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        return img
    
    height, width, _ = image.shape
    area = height * width
    image = calArea(image, area)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    gray = clahe.apply(gray)
    # gray = cv2.equalizeHist(gray)
    kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    gray = cv2.convertScaleAbs(gray, alpha=1, beta=50)

    inverted = cv2.bitwise_not(gray)

    return inverted

# %%
def ocr(image, entity_name):
    def process_inference_output(texts, valid_units):
        processed_texts = []

        for text in texts:
            # Ensure space after each number
            text_with_spaces = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
            
            # Split words separated by slashes, commas, or periods
            words = re.split(r'[\/,;]', text_with_spaces)

            # Add processed words to the list
            for word in words:
                cleaned_word = word.strip()
                if cleaned_word:  # Ensure the word is not empty
                    processed_texts.append(cleaned_word)
        final_texts = []
        for text in processed_texts:
            # Check if the text contains a valid unit
            for unit in valid_units:
                if unit in text:
                    # Find the position of the last character of the valid unit
                    unit_pos = text.find(unit) + len(unit)
                    # Slice the text to keep only the valid unit and the preceding number
                    cleaned_text = text[:unit_pos].strip()
                    final_texts.append(cleaned_text)
                    break
            else:
                # If no valid unit is found, keep the original text
                final_texts.append(text)
        
        return final_texts
    
    def filter_invalid_elements(texts, valid_units):
        filtered_texts = []

        for text in texts:
            words = text.split()
            is_valid = True
            for word in words:
                if word not in valid_units and not re.match(r'\d+(\.\d+)?', word):
                    is_valid = False
                    break
            if is_valid:
                filtered_texts.append(text)
        
        return filtered_texts

    img = preprocess(image)

    result = reader.readtext(img)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    unique_values = set()
    valid_units = entity_unit_map[entity_name]
    cleaned_texts = []
    for detection in result:
        text = detection[1]
        
        text_no_spaces = re.sub(r'(\d)\s+', r'\1', text)
    
        # Step 2: Process the text word by word
        words = text_no_spaces.split()
        processed_words = []
        for word in words:
        # If the word contains a digit, keep it; otherwise, discard it
            if re.search(r'\d', word):
                processed_words.append(word)
    
        # Join the processed words back into a single string
        final_text = ' '.join(processed_words)
        
        # Replace double quotes with ' inches'
        final_text = final_text.replace('\"', ' in')
        
        # Remove specific unwanted characters
        final_text = final_text.replace('*', '').replace(':', '').replace('+', '')
        
        # Remove sequences of 6 or more digits
        final_text = re.sub(r'\b\d{6,}\b', '', final_text)
        
        # Remove sequences of 6 or more digits followed by alphabetic characters
        final_text = re.sub(r'\b\d{6,}[a-zA-Z]*\b', '', final_text)
        
        # Remove sequences of digits separated by spaces
        final_text = re.sub(r'\b\d+\s+\d+\b', '', final_text)
        
        # Remove sequences of digits followed by an asterisk
        final_text = re.sub(r'\b\d+\*\b', '', final_text)
        
        # Remove text within square or round brackets
        final_text = re.sub(r'\[.*?\]|\(.*?\)', '', final_text)
        
        # Remove sequences of digits separated by two dots
        final_text = re.sub(r'\b\d+\.\.\d+\b', '', final_text)
        
        # Remove sequences of digits preceded by whitespace
        final_text = re.sub(r'\s+\d+', '', final_text)

        # Remove trailing special characters
        final_text = re.sub(r'[^a-zA-Z0-9\s]+$', '', final_text)

        # Replace any occurrence of a number followed by a point, followed by 'S', and then followed by another number with '5' instead of 'S'
        final_text = re.sub(r'(?<=\d|\.)S(?=\d|\.)', '5', final_text)
        cleaned_texts.append(final_text)
    
    
    for i in range(len(cleaned_texts)):
        contains_valid_unit = any(unit in cleaned_texts[i] for unit in valid_units)
        if not contains_valid_unit:
            for abbr, full_form in abbreviation_map.items():
                if abbr in cleaned_texts[i] and full_form in valid_units:
                    cleaned_texts[i] = cleaned_texts[i].replace(abbr, full_form)
                    # print(f"Replaced '{abbr}' with '{full_form}': {cleaned_texts[i]}")
                    break
    cleaned_texts = process_inference_output(cleaned_texts,valid_units)
    # Filter out unique valid texts
    final_unique_texts = []
    unique_values = set()
    # print(cleaned_texts)
    for text in cleaned_texts:
        normalized_text = text.strip().lower()  # Normalize the text by stripping whitespace and converting to lowercase
        if normalized_text not in unique_values:
            for unit in valid_units:
                if unit in normalized_text:
                    unique_values.add(normalized_text)
                    final_unique_texts.append(text)  # Append the original text
                    break
    # print(final_unique_texts)

    final_unique_texts = filter_invalid_elements(final_unique_texts, valid_units)

    return final_unique_texts

# %%
def prediction(image, category_id, entity_name):
    def convert_to_common_unit(value, entity_name):
        match = re.match(r"([0-9.]+)\s*(\w+)", value)
        if not match:
            return None
        num, unit = match.groups()
        num = float(num)
        unit = unit.lower()
        
        if entity_name in ['width', 'depth', 'height']:
            if unit in ['mm', 'millimeter', 'millimeters', 'millimetre', 'millimetres']:
                return num / 1000
            elif unit in ['cm', 'centimeter', 'centimeters', 'centimetre']:
                return num / 100
            elif unit in ['m', 'meter', 'meters', 'metre', 'metres']:
                return num
            elif unit in ['km', 'kilometer', 'kilometers', 'kilometre', 'kilometres']:
                return num * 1000
            elif unit in ['ft', 'foot', 'feet']:
                return num * 0.3048
            elif unit in ['in', 'inch', 'inches']:
                return num * 0.0254
            elif unit in ['yd', 'yard', 'yards']:
                return num * 0.9144
        elif entity_name in ['item_weight', 'maximum_weight_recommendation']:
            if unit in ['mg', 'milligram', 'milligrams']:
                return num / 1000
            elif unit in ['g', 'gram', 'grams']:
                return num
            elif unit in ['kg', 'kilogram', 'kilograms']:
                return num * 1000
            elif unit in ['t', 'ton', 'tons']:
                return num * 1e6
            elif unit in ['oz', 'ounce', 'ounces']:
                return num * 28.3495
            elif unit in ['lb', 'pound', 'pounds']:
                return num * 453.592
        elif entity_name == 'voltage':
            if unit in ['mv', 'millivolt', 'millivolts']:
                return num / 1000
            elif unit in ['kv', 'kilovolt', 'kilovolts']:
                return num * 1000
            elif unit in ['v', 'volt', 'volts']:
                return num
        elif entity_name == 'wattage':
            if unit in ['kw', 'kilowatt', 'kilowatts']:
                return num * 1000
            elif unit in ['w', 'watt', 'watts']:
                return num
        elif entity_name == 'item_volume':
            if unit in ['ml', 'millilitre', 'millilitres', 'milliliter', 'milliliters']:
                return num
            elif unit in ['l', 'litre', 'litres', 'liter', 'liters']:
                return num * 1000
            elif unit in ['cl', 'centilitre', 'centilitres', 'centiliter', 'centiliters']:
                return num * 10
            elif unit in ['dl', 'decilitre', 'decilitres', 'deciliter', 'deciliters']:
                return num * 100
            elif unit in ['fl oz', 'fluid ounce', 'fluid ounces']:
                return num * 29.5735
            elif unit in ['cup', 'cups']:
                return num * 240
            elif unit in ['pt', 'pint', 'pints']:
                return num * 473.176
            elif unit in ['qt', 'quart', 'quarts']:
                return num * 946.353
            elif unit in ['gal', 'gallon', 'gallons']:
                return num * 3785.41
            elif unit in ['imp gal', 'imperial gallon', 'imperial gallons']:
                return num * 4546.09
            elif unit in ['cu ft', 'cubic foot', 'cubic feet']:
                return num * 28316.8
            elif unit in ['cu in', 'cubic inch', 'cubic inches']:
                return num * 16.3871
        return None
    
    def get_dimension(model, group_id, entity_name, entity_values):
        group_id_encoded = label_encoder_group.transform([group_id])[0]
        entity_name_encoded = label_encoder_entity.transform([entity_name])[0]
        
        best_entity_value = None
        highest_score = -1
        
        for entity_value in entity_values:
            entity_value_meters = convert_to_common_unit(entity_value, entity_name)
            if entity_value_meters is None:
                continue
            input_data = np.array([[group_id_encoded, entity_value_meters]])
            
            # Get confidence scores for all classes
            confidence_scores = model.predict_proba(input_data)[0]
            
            # Get the score for the specific entity_name
            score = confidence_scores[entity_name_encoded]
            
            if score > highest_score:
                highest_score = score
                best_entity_value = entity_value
        
        if best_entity_value is None:
            raise ValueError("No valid entity_value found")
        
        return best_entity_value
    
    text = ocr(image, entity_name)
    # print("OCR: ", text)
    if text:
        if entity_name not in ['voltage', 'item_weight', 'maximum_weight_recommendation', 'wattage']:
            value = get_dimension(loaded_model, category_id, entity_name, text)
        else: 
            value = text[0]
    else:
        value = ""
    return value

def process_row(row):
    return predictor(row['image_link'], row['group_id'], row['entity_name'])


# %%
def predictor(image_link, category_id, entity_name):
    '''
    Call your model/approach here
    '''
    def extract_filename(url):
        # Regular expression to match the file name before .jpg
        match = re.search(r'/([^/]+)\.jpg$', url)
        if match:
            return match.group(1)
        return None
    filename = f'{extract_filename(image_link)}.jpg'
    image_path = os.path.join(r'../images/test', filename)
    img = cv2.imread(image_path)
    # print(img.shape)
    # print("Entity Name: ", entity_name)
    try:
        return prediction(img, category_id, entity_name)
    except:
        return ""
  
    

# %%
if __name__ == "__main__":
    DATASET_FOLDER = '../dataset/'
    
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Initialize the prediction column with empty strings
    test['prediction'] = ""
    
    # Process rows in chunks and save intermediate results
    for start in tqdm(range(0, len(test), 100), desc="Processing rows"):
        end = min(start + 100, len(test))
        test.loc[start:end-1, 'prediction'] = test.loc[start:end-1].apply(
            lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
        
        # Save intermediate results
        output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
        test[['index', 'prediction']].to_csv(output_filename, index=False)
    
    # Save final results
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)


