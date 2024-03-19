from flask import Flask, render_template, request
import PyPDF2
import os

app = Flask(__name__)


from PyPDF2 import PdfReader
import tempfile
# UPLOAD_FOLDER = 'uploads'
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------

# my custome code to extract file OCR
import fitz 
from PIL import Image
import pytesseract
import io
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import spacy
import re
import json
from textblob import TextBlob
import random
import spacy
from nltk.corpus import wordnet
from collections import Counter


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')




para = ""
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    
    extracted_text = ""

    for page_number in range(pdf_document.page_count):

        page = pdf_document[page_number]

        page_text = page.get_text()

        extracted_text += page_text

    pdf_document.close()

    return extracted_text

def ocr_with_tesseract(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

def extract_text_code(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)


    with open("extracted_text.txt", "w", encoding="utf-8") as text_file:
        text_file.write(pdf_text)

    image_dir = "images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Perform OCR on each page using Tesseract
    pdf_document = fitz.open(pdf_path)  
    for page_number in range(pdf_document.page_count):
        image_path = os.path.join(image_dir, f"page_{page_number + 1}.png")
        pixmap = pdf_document[page_number].get_pixmap()


        img_data = pixmap.tobytes("png")

        # Create a PIL Image from the PNG data:
        img = Image.open(io.BytesIO(img_data))  

        img.save(image_path)

        # Perform OCR on the image using Tesseract
        ocr_result = ocr_with_tesseract(image_path)
        # print(f"Text from page {page_number + 1}:\n{ocr_result}")
        global para
        para +=ocr_result

    pdf_document.close()

def OCR_code(case_info):
    pdf_path = case_info["file_path"]
    extract_text_code(pdf_path)



def summarize_document(full_text, num_sentences=3):
    document_text = full_text


    sentences = sent_tokenize(document_text)


    words = word_tokenize(document_text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    # Calculate word frequencies
    word_freq = FreqDist(filtered_words)

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence):
            if word.lower() in word_freq:
                if i not in sentence_scores:
                    sentence_scores[i] = word_freq[word.lower()]
                else:
                    sentence_scores[i] += word_freq[word.lower()]

    # Select the top sentences based on scores
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    # Sort selected sentences by their original order
    top_sentences.sort()

    # Generate the summarized text
    summarized_text = ' '.join([sentences[i] for i in top_sentences])

    return summarized_text




def process_document(full_text):
    nlp = spacy.load("en_core_web_lg")

    document_text = full_text

    # Process the document using spaCy
    doc = nlp(document_text)


    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]


    processed_text = ' '.join(lemmatized_tokens)

    return processed_text




































# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------







# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------




# Define the JSON structure
data = {
    "a": {
        "aa": {},
        "ab": {},
        "ac": {},
        "ad": {},
        "ae": {},
        "af": {},
        "ag": {},
        "ah": {},
        "ai": {},
        "aj": {},
        "ak": {},
        "al": {},
        "am": {},
        "an": {},
        "ao": {},
        "ap": {},
        "aq": {},
        "ar": {},
        "as": {},
        "at": {},
        "au": {},
        "av": {},
        "aw": {},
        "ax": {},
        "ay": {},
        "az": {}
    },
    "b": {
        "ba": {},
        "bb": {},
        "bc": {},
        "bd": {},
        "be": {},
        "bf": {},
        "bg": {},
        "bh": {},
        "bi": {},
        "bj": {},
        "bk": {},
        "bl": {},
        "bm": {},
        "bn": {},
        "bo": {},
        "bp": {},
        "bq": {},
        "br": {},
        "bs": {},
        "bt": {},
        "bu": {},
        "bv": {},
        "bw": {},
        "bx": {},
        "by": {},
        "bz": {}
    },
    "c": {
        "ca": {},
        "cb": {},
        "cc": {},
        "cd": {},
        "ce": {},
        "cf": {},
        "cg": {},
        "ch": {},
        "ci": {},
        "cj": {},
        "ck": {},
        "cl": {},
        "cm": {},
        "cn": {},
        "co": {},
        "cp": {},
        "cq": {},
        "cr": {},
        "cs": {},
        "ct": {},
        "cu": {},
        "cv": {},
        "cw": {},
        "cx": {},
        "cy": {},
        "cz": {}
    },
    "d": {
        "da": {},
        "db": {},
        "dc": {},
        "dd": {},
        "de": {},
        "df": {},
        "dg": {},
        "dh": {},
        "di": {},
        "dj": {},
        "dk": {},
        "dl": {},
        "dm": {},
        "dn": {},
        "do": {},
        "dp": {},
        "dq": {},
        "dr": {},
        "ds": {},
        "dt": {},
        "du": {},
        "dv": {},
        "dw": {},
        "dx": {},
        "dy": {},
        "dz": {}
    },
    "e": {
        "ea": {},
        "eb": {},
        "ec": {},
        "ed": {},
        "ee": {},
        "ef": {},
        "eg": {},
        "eh": {},
        "ei": {},
        "ej": {},
        "ek": {},
        "el": {},
        "em": {},
        "en": {},
        "eo": {},
        "ep": {},
        "eq": {},
        "er": {},
        "es": {},
        "et": {},
        "eu": {},
        "ev": {},
        "ew": {},
        "ex": {},
        "ey": {},
        "ez": {}
    },
    "f": {
        "fa": {},
        "fb": {},
        "fc": {},
        "fd": {},
        "fe": {},
        "ff": {},
        "fg": {},
        "fh": {},
        "fi": {},
        "fj": {},
        "fk": {},
        "fl": {},
        "fm": {},
        "fn": {},
        "fo": {},
        "fp": {},
        "fq": {},
        "fr": {},
        "fs": {},
        "ft": {},
        "fu": {},
        "fv": {},
        "fw": {},
        "fx": {},
        "fy": {},
        "fz": {}
    },
    "g": {
        "ga": {},
        "gb": {},
        "gc": {},
        "gd": {},
        "ge": {},
        "gf": {},
        "gg": {},
        "gh": {},
        "gi": {},
        "gj": {},
        "gk": {},
        "gl": {},
        "gm": {},
        "gn": {},
        "go": {},
        "gp": {},
        "gq": {},
        "gr": {},
        "gs": {},
        "gt": {},
        "gu": {},
        "gv": {},
        "gw": {},
        "gx": {},
        "gy": {},
        "gz": {}
    },
    "h": {
        "ha": {},
        "hb": {},
        "hc": {},
        "hd": {},
        "he": {},
        "hf": {},
        "hg": {},
        "hh": {},
        "hi": {},
        "hj": {},
        "hk": {},
        "hl": {},
        "hm": {},
        "hn": {},
        "ho": {},
        "hp": {},
        "hq": {},
        "hr": {},
        "hs": {},
        "ht": {},
        "hu": {},
        "hv": {},
        "hw": {},
        "hx": {},
        "hy": {},
        "hz": {}
    },
    "i": {
        "ia": {},
        "ib": {},
        "ic": {},
        "id": {},
        "ie": {},
        "if": {},
        "ig": {},
        "ih": {},
        "ii": {},
        "ij": {},
        "ik": {},
        "il": {},
        "im": {},
        "in": {},
        "io": {},
        "ip": {},
        "iq": {},
        "ir": {},
        "is": {},
        "it": {},
        "iu": {},
        "iv": {},
        "iw": {},
        "ix": {},
        "iy": {},
        "iz": {}
    },
    "j": {
        "ja": {},
        "jb": {},
        "jc": {},
        "jd": {},
        "je": {},
        "jf": {},
        "jg": {},
        "jh": {},
        "ji": {},
        "jj": {},
        "jk": {},
        "jl": {},
        "jm": {},
        "jn": {},
        "jo": {},
        "jp": {},
        "jq": {},
        "jr": {},
        "js": {},
        "jt": {},
        "ju": {},
        "jv": {},
        "jw": {},
        "jx": {},
        "jy": {},
        "jz": {}
    },
    "k": {
        "ka": {},
        "kb": {},
        "kc": {},
        "kd": {},
        "ke": {},
        "kf": {},
        "kg": {},
        "kh": {},
        "ki": {},
        "kj": {},
        "kk": {},
        "kl": {},
        "km": {},
        "kn": {},
        "ko": {},
        "kp": {},
        "kq": {},
        "kr": {},
        "ks": {},
        "kt": {},
        "ku": {},
        "kv": {},
        "kw": {},
        "kx": {},
        "ky": {},
        "kz": {}
    },
    "l": {
        "la": {},
        "lb": {},
        "lc": {},
        "ld": {},
        "le": {},
        "lf": {},
        "lg": {},
        "lh": {},
        "li": {},
        "lj": {},
        "lk": {},
        "ll": {},
        "lm": {},
        "ln": {},
        "lo": {},
        "lp": {},
        "lq": {},
        "lr": {},
        "ls": {},
        "lt": {},
        "lu": {},
        "lv": {},
        "lw": {},
        "lx": {},
        "ly": {},
        "lz": {}
    },
    "m": {
        "ma": {},
        "mb": {},
        "mc": {},
        "md": {},
        "me": {},
        "mf": {},
        "mg": {},
        "mh": {},
        "mi": {},
        "mj": {},
        "mk": {},
        "ml": {},
        "mm": {},
        "mn": {},
        "mo": {},
        "mp": {},
        "mq": {},
        "mr": {},
        "ms": {},
        "mt": {},
        "mu": {},
        "mv": {},
        "mw": {},
        "mx": {},
        "my": {},
        "mz": {}
    },
    "n": {
        "na": {},
        "nb": {},
        "nc": {},
        "nd": {},
        "ne": {},
        "nf": {},
        "ng": {},
        "nh": {},
        "ni": {},
        "nj": {},
        "nk": {},
        "nl": {},
        "nm": {},
        "nn": {},
        "no": {},
        "np": {},
        "nq": {},
        "nr": {},
        "ns": {},
        "nt": {},
        "nu": {},
        "nv": {},
        "nw": {},
        "nx": {},
        "ny": {},
        "nz": {}
    },
    "o": {
        "oa": {},
        "ob": {},
        "oc": {},
        "od": {},
        "oe": {},
        "of": {},
        "og": {},
        "oh": {},
        "oi": {},
        "oj": {},
        "ok": {},
        "ol": {},
        "om": {},
        "on": {},
        "oo": {},
        "op": {},
        "oq": {},
        "or": {},
        "os": {},
        "ot": {},
        "ou": {},
        "ov": {},
        "ow": {},
        "ox": {},
        "oy": {},
        "oz": {}
    },
    "p": {
        "pa": {},
        "pb": {},
        "pc": {},
        "pd": {},
        "pe": {},
        "pf": {},
        "pg": {},
        "ph": {},
        "pi": {},
        "pj": {},
        "pk": {},
        "pl": {},
        "pm": {},
        "pn": {},
        "po": {},
        "pp": {},
        "pq": {},
        "pr": {},
        "ps": {},
        "pt": {},
        "pu": {},
        "pv": {},
        "pw": {},
        "px": {},
        "py": {},
        "pz": {}
    },
    "q": {
        "qa": {},
        "qb": {},
        "qc": {},
        "qd": {},
        "qe": {},
        "qf": {},
        "qg": {},
        "qh": {},
        "qi": {},
        "qj": {},
        "qk": {},
        "ql": {},
        "qm": {},
        "qn": {},
        "qo": {},
        "qp": {},
        "qq": {},
        "qr": {},
        "qs": {},
        "qt": {},
        "qu": {},
        "qv": {},
        "qw": {},
        "qx": {},
        "qy": {},
        "qz": {}
    },
    "r": {
        "ra": {},
        "rb": {},
        "rc": {},
        "rd": {},
        "re": {},
        "rf": {},
        "rg": {},
        "rh": {},
        "ri": {},
        "rj": {},
        "rk": {},
        "rl": {},
        "rm": {},
        "rn": {},
        "ro": {},
        "rp": {},
        "rq": {},
        "rr": {},
        "rs": {},
        "rt": {},
        "ru": {},
        "rv": {},
        "rw": {},
        "rx": {},
        "ry": {},
        "rz": {}
    },
    "s": {
        "sa": {},
        "sb": {},
        "sc": {},
        "sd": {},
        "se": {},
        "sf": {},
        "sg": {},
        "sh": {},
        "si": {},
        "sj": {},
        "sk": {},
        "sl": {},
        "sm": {},
        "sn": {},
        "so": {},
        "sp": {},
        "sq": {},
        "sr": {},
        "ss": {},
        "st": {},
        "su": {},
        "sv": {},
        "sw": {},
        "sx": {},
        "sy": {},
        "sz": {}
    },
    "t": {
        "ta": {},
        "tb": {},
        "tc": {},
        "td": {},
        "te": {},
        "tf": {},
        "tg": {},
        "th": {},
        "ti": {},
        "tj": {},
        "tk": {},
        "tl": {},
        "tm": {},
        "tn": {},
        "to": {},
        "tp": {},
        "tq": {},
        "tr": {},
        "ts": {},
        "tt": {},
        "tu": {},
        "tv": {},
        "tw": {},
        "tx": {},
        "ty": {},
        "tz": {}
    },
    "u": {
        "ua": {},
        "ub": {},
        "uc": {},
        "ud": {},
        "ue": {},
        "uf": {},
        "ug": {},
        "uh": {},
        "ui": {},
        "uj": {},
        "uk": {},
        "ul": {},
        "um": {},
        "un": {},
        "uo": {},
        "up": {},
        "uq": {},
        "ur": {},
        "us": {},
        "ut": {},
        "uu": {},
        "uv": {},
        "uw": {},
        "ux": {},
        "uy": {},
        "uz": {}
    },
    "v": {
        "va": {},
        "vb": {},
        "vc": {},
        "vd": {},
        "ve": {},
        "vf": {},
        "vg": {},
        "vh": {},
        "vi": {},
        "vj": {},
        "vk": {},
        "vl": {},
        "vm": {},
        "vn": {},
        "vo": {},
        "vp": {},
        "vq": {},
        "vr": {},
        "vs": {},
        "vt": {},
        "vu": {},
        "vv": {},
        "vw": {},
        "vx": {},
        "vy": {},
        "vz": {}
    },
    "w": {
        "wa": {},
        "wb": {},
        "wc": {},
        "wd": {},
        "we": {},
        "wf": {},
        "wg": {},
        "wh": {},
        "wi": {},
        "wj": {},
        "wk": {},
        "wl": {},
        "wm": {},
        "wn": {},
        "wo": {},
        "wp": {},
        "wq": {},
        "wr": {},
        "ws": {},
        "wt": {},
        "wu": {},
        "wv": {},
        "ww": {},
        "wx": {},
        "wy": {},
        "wz": {}
    },
    "x": {
        "xa": {},
        "xb": {},
        "xc": {},
        "xd": {},
        "xe": {},
        "xf": {},
        "xg": {},
        "xh": {},
        "xi": {},
        "xj": {},
        "xk": {},
        "xl": {},
        "xm": {},
        "xn": {},
        "xo": {},
        "xp": {},
        "xq": {},
        "xr": {},
        "xs": {},
        "xt": {},
        "xu": {},
        "xv": {},
        "xw": {},
        "xx": {},
        "xy": {},
        "xz": {}
    },
    "y": {
        "ya": {},
        "yb": {},
        "yc": {},
        "yd": {},
        "ye": {},
        "yf": {},
        "yg": {},
        "yh": {},
        "yi": {},
        "yj": {},
        "yk": {},
        "yl": {},
        "ym": {},
        "yn": {},
        "yo": {},
        "yp": {},
        "yq": {},
        "yr": {},
        "ys": {},
        "yt": {},
        "yu": {},
        "yv": {},
        "yw": {},
        "yx": {},
        "yy": {},
        "yz": {}
    },
    "z": {
        "za": {},
        "zb": {},
        "zc": {},
        "zd": {},
        "ze": {},
        "zf": {},
        "zg": {},
        "zh": {},
        "zi": {},
        "zj": {},
        "zk": {},
        "zl": {},
        "zm": {},
        "zn": {},
        "zo": {},
        "zp": {},
        "zq": {},
        "zr": {},
        "zs": {},
        "zt": {},
        "zu": {},
        "zv": {},
        "zw": {},
        "zx": {},
        "zy": {},
        "zz": {}
    }
}







def save_to_json_local(case_info,prior,priortext):
  file_path_json = 'data.json'
  with open(file_path_json, 'r') as file:
    dataImport = json.load(file)

  words = case_info["compressed_text"].split()
  for word in words:
    if len(word) >= 2:
      if word[0].lower().isalnum() and word[1].lower().isalnum(): 
        first_two_letters = word[0].lower() + word[1].lower()
        if word[0].lower() in dataImport:
          if word in dataImport[word[0].lower()][first_two_letters]:
            # data[word[0].lower()][first_two_letters][word]["file_path"]=list(set(data[word[0].lower()][first_two_letters][word]["file_path"].append(case_info["file_path"])))
            # Append case_info["file_path"] to the list
            dataImport[word[0].lower()][first_two_letters][word]["file_path"].append(case_info["file_path"])

# Convert the list to a set to remove duplicates and then back to a list
            dataImport[word[0].lower()][first_two_letters][word]["file_path"] = list(set(dataImport[word[0].lower()][first_two_letters][word]["file_path"]))


            dataImport[word[0].lower()][first_two_letters][word]["sections"]=list(set(dataImport[word[0].lower()][first_two_letters][word]["sections"]).union(set(case_info["sections"])))

            dataImport[word[0].lower()][first_two_letters][word]["acts"]=list(set(dataImport[word[0].lower()][first_two_letters][word]["acts"]).union(set(case_info["acts"])))
            
          else:
              dataImport[word[0].lower()][first_two_letters][word] = {"priority":prior,"file_path":[case_info["file_path"]],"sections":case_info["sections"],"acts":case_info["acts"]}
        else:
            print(f"First two letters '{first_two_letters}' not found in the JSON structure.")
  with open(file_path_json, 'w') as file:
    json.dump(dataImport, file, indent=4)         
  # json_dataImport = json.dumps(data, indent=4)
  # file_path_json = 'data.json'
  # with open(file_path_json, 'w') as file:
  #   json.dump(data, file)

  













def save_to_json_local2(case_info,prior,priortext):
  file_path_json = 'data.json'
  with open(file_path_json, 'r') as file:
    dataImport = json.load(file)

  words = priortext.split(',')
  for word in words:
    if len(word) >= 2:
      if word[0].lower().isalnum() and word[1].lower().isalnum(): 
        first_two_letters = word[0].lower() + word[1].lower()
        if word[0].lower() in dataImport:
          if word in dataImport[word[0].lower()][first_two_letters]:
            # data[word[0].lower()][first_two_letters][word]["file_path"]=list(set(data[word[0].lower()][first_two_letters][word]["file_path"].append(case_info["file_path"])))
            # Append case_info["file_path"] to the list
            dataImport[word[0].lower()][first_two_letters][word]["file_path"].append(case_info["file_path"])

# Convert the list to a set to remove duplicates and then back to a list
            dataImport[word[0].lower()][first_two_letters][word]["file_path"] = list(set(dataImport[word[0].lower()][first_two_letters][word]["file_path"]))


            dataImport[word[0].lower()][first_two_letters][word]["sections"]=list(set(dataImport[word[0].lower()][first_two_letters][word]["sections"]).union(set(case_info["sections"])))

            dataImport[word[0].lower()][first_two_letters][word]["acts"]=list(set(dataImport[word[0].lower()][first_two_letters][word]["acts"]).union(set(case_info["acts"])))
            
          else:
              dataImport[word[0].lower()][first_two_letters][word] = {"priority":prior,"file_path":[case_info["file_path"]],"sections":case_info["sections"],"acts":case_info["acts"]}
        else:
            print(f"First two letters '{first_two_letters}' not found in the JSON structure.")
  with open(file_path_json, 'w') as file:
    json.dump(dataImport, file, indent=4)         
  # json_dataImport = json.dumps(data, indent=4)
  # file_path_json = 'data.json'
  # with open(file_path_json, 'w') as file:
  #   json.dump(data, file)











# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------

# def extract_text_from_pdf(file):
#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        
#         file.save(temp_file)
#         with open(temp_file.name, 'rb') as pdf_file:
#             pdf_reader = PdfReader(pdf_file)
#             text = ''
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#     return text



@app.route('/', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        # first_name = request.form['first_name']
        # last_name = request.form['last_name']
        # case_id = 
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        if file:
          file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
          file.save(file_path)
          # ctest = {
          #    "file_path": 'test',
          #    "court_name": None,
    
          #     }
          # return render_template('summary.html', data=ctest)
          data = {
            "key1": "value1",
             "key2": "value2"
           }
          print(data)
          return render_template('summary.html', data=data)

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------

          case_info = {
             "file_path": file_path,
             "court_name": None,
    
             "case_id": 0,

             "case_year": 0,
    
             "judges_names": [],
    
             "parties_involved": {
                 "appellants": [],
                 "respondents": []
             },
    
              "sections": [],
    
              "acts": [],
    
             "summary":None,
    
              "compressed_text": None,
    
             "similar_cases": []
              }

          case_info['court_name'] = request.form['court_name']
          case_info['case_id'] = request.form['case_id']
          case_info['case_year'] = request.form['case_year']
          case_info['judges_names'] = request.form['judges_names'].split()
          case_info['parties_involved']['appellants'] = request.form['appellants'].split()
          case_info['parties_involved']['respondents'] = request.form['respondents'].split()

          case_info_str = f"Name of Court = '{case_info['court_name']}'<br>" \
              f"Case ID = {case_info['case_id']}<br>" \
              f"Case Year = {case_info['case_year']}<br>" \
              f"Jydges Names= {request.form['judges_names']}<br>" \
              f"Appellants Name = {request.form['appellants']}<br>" \
              f"Respondents Name = {request.form['respondents']}<br>"
           


            


          OCR_code(case_info)
          summary = summarize_document(para)
          case_info["summary"] = case_info_str + " \n " + summary
          processed_text = process_document(para)
          case_info["compressed_text"] = processed_text




          text = case_info["compressed_text"]


          section_pattern = re.compile(r'\bSection \d+\b', re.IGNORECASE)
          section_to_act_pattern = re.compile(r'\bSection\b\s+(\w+\s+){2,3}\bAct\b',re.DOTALL | re.IGNORECASE)

          # Find all matches in the text
          sections = section_pattern.findall(text)
          acts = section_to_act_pattern.findall(text)

          # Print the results and store 
          sections = [word.lower() for word in sections]
          sections = list(set(sections))
          acts = [word.lower() for word in acts]
          acts = list(set(acts))
          case_info["acts"] = acts


          case_info["sections"] = sections



          file_path_json = 'filesdata.json'
          with open(file_path_json, 'r') as file:
            dataFileImport = json.load(file)


          
          
          dataFileImport[case_info["file_path"]] = case_info 
          with open(file_path_json, 'w') as file:
            json.dump(dataFileImport, file, indent=4) 








          prior = 1
          priortext = f"{case_info['file_path']},{case_info['court_name']},{case_info['case_id']},{case_info['case_year']},{' ,'.join(case_info['judges_names'])},{' ,'.join(case_info['parties_involved']['appellants'])},{' ,'.join(case_info['parties_involved']['respondents'])} "
          save_to_json_local2(case_info,prior,priortext)
          prior = 2
          priortext = ''
          save_to_json_local(case_info,prior,priortext)
 # -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
            # pdf_text = extract_text_from_pdf(file)
            # pdf_text = "hd fj"
            # num_words = len(pdf_text.split())
          print(case_info)
          ctest = {
             "file_path": 'test',
             "court_name": 'None',
    
              }
          return render_template('summary.html', data = ctest)
    
    return render_template('index.html')





def get_synonyms(word):
    synonyms = []

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())

    return list(set(synonyms))  # Convert to set to remove duplicates, then back to list

def extract_words(paragraph):
    words = word_tokenize(paragraph)
    return [word.lower() for word in words if word.isalnum()]  # Remove non-alphabetic characters

def calculate_match_percentage(json_data, sections_list, acts_list):
    filesList = {}


    # for key, value in json_data.items():
    for inner_key, inner_value in json_data.items():
        sections = inner_value["sections"]
        acts = inner_value["acts"]
        total_sections = max(len(sections_list), 1) 
        total_acts = max(len(acts_list), 1) 
        if total_sections == 0 or total_acts == 0:
          pass
        matched_sections = sum(1 for section in sections if section in sections_list) or 1
        matched_acts = sum(1 for act in acts if act in acts_list) or 1
        section_match_percentage = (matched_sections / total_sections) * 100 or 1
        act_match_percentage = (matched_acts / total_acts) * 100  or 1
        total_match_percentage = (section_match_percentage + act_match_percentage) / 2
        if total_match_percentage > 50:
          filesList[inner_key] = total_match_percentage

    sorted_listFile = [key for key, _ in sorted(filesList.items(), key=lambda x: -x[1])]

    return sorted_listFile

    
    
    
    
    

    
   
    
   
    
    
      


    

def search_synonyms_in_json(synonyms_dict):
  file_path_json = 'data.json'
  with open(file_path_json, 'r') as file:
    dataImport = json.load(file)
  file_path_json = 'filesdata.json'
  with open(file_path_json, 'r') as file:
    filedataImport = json.load(file)

  one_d_list = []
  for key, value in synonyms_dict.items():
    if isinstance(value, list):
      one_d_list.extend([key, *value])
    else:
      one_d_list.extend([key, value])
  temp_dest = { }
   




  for word in one_d_list:
    if len(word) >= 2:
      if word[0].lower().isalnum() and word[1].lower().isalnum(): 
        first_two_letters = word[0].lower() + word[1].lower()
        if word[0].lower() in dataImport:
          if word in dataImport[word[0].lower()][first_two_letters]:
            temp_dest[word] =  dataImport[word[0].lower()][first_two_letters][word]

  file_path_counts = Counter()
  file_path_priorities = Counter()
  unique_sections = set()
  unique_acts = set()
# Iterate over the data and accumulate counts and priorities
  # for key, value in data.items():
  for key, value in temp_dest.items():
    unique_sections.update(value['sections'])
    unique_acts.update(value['acts'])
    for file_path in value['file_path']:
      file_path_counts[file_path] += 1
      file_path_priorities[file_path] += value['priority']


    # for file_path, priority in zip(value['file_path'], [value['priority']] * len(value['file_path'])):
    #   file_path_counts[file_path] += 1
    #   file_path_priorities[file_path] += priority
  temp_files_with_value = {}
  totalCP = 0
  countFiles = 0
# Print the sums of count and priority for each file path
  for file_path in file_path_counts.keys():
      total_count = file_path_counts[file_path]
      total_priority = file_path_priorities[file_path]
      totalCandP = total_count + total_priority*10
      temp_files_with_value[file_path] = totalCandP
      
      totalCP += totalCandP
      countFiles +=1
  

  # sorted_List_wordMatch = [key for key, _ in sorted(temp_files_with_value.items(), key=lambda x: -x[1])]

  avgCandp = totalCP/countFiles

  sorted_List_wordMatch = {key: value for key, value in sorted(temp_files_with_value.items(), key=lambda x: x[1], reverse = True) if value >= avgCandp}
  sorted_List_wordMatch = list(sorted_List_wordMatch.keys())
  




  filesListSecAct = calculate_match_percentage(filedataImport, list(unique_sections),list(unique_acts))



  finalList = sorted_List_wordMatch + filesListSecAct
  ordered_set = set()
  for item in finalList:
    ordered_set.add(item)

  return list(ordered_set)


# ------------------------------------------------------

    
    

@app.route('/process_query', methods = ['GET','POST'])
def process_query():
    query = request.form['query']


    # Process the query here (e.g., search for the query in the PDF file)
    blob = TextBlob(query)
    corrected_text = str(blob.correct())
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(corrected_text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]
    query_processed_text = ' '.join(lemmatized_tokens)
    input_words = extract_words(query_processed_text)

    synonyms_dict = {}
    for word in input_words:
      synonyms_dict[word] = get_synonyms(word)
    
    
    dic_with_val = search_synonyms_in_json(synonyms_dict)

    
 
    




    # return f"Query: {dic_with_val}"
    return render_template('file_list.html', files=dic_with_val)

















if __name__ == '__main__':
    app.run(debug=True)
