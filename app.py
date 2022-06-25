import os
os.system("pip install transformers")
os.system("pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f "
          "https://download.pytorch.org/whl/cpu/torch_stable.html")
os.system("pip install mtranslate")
os.system("pip install requests")
os.system("pip install random")

import transformers
import json
import random
import requests

from mtranslate import translate
import streamlit as st


MODELS = {
    "GPT-2 Model Recycled From English": {
        "url": "https://api-inference.huggingface.co/models/GroNLP/gpt2-small-dutch"
    },
}

PROMPT_LIST = {
    "Er was eens...": ["Er was eens..."],
    "Dag.": ["Hallo, mijn naam is "],
    "Te zijn of niet te zijn?": ["Naar mijn mening is 'zijn'"],
}


def query(payload, model_name):
    data = json.dumps(payload)
    print("model url:", MODELS[model_name]["url"])
    response = requests.request(
        "POST", MODELS[model_name]["url"], headers={}, data=data
    )
    return json.loads(response.content.decode("utf-8"))


def process(
    text: str, model_name: str, max_len: int, temp: float, top_k: int, top_p: float
):
    payload = {
        "inputs": text,
        "parameters": {
            "max_new_tokens": max_len,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temp,
            "repetition_penalty": 2.0,
        },
        "options": {
            "use_cache": True,
        },
    }
    return query(payload, model_name)


# Page
st.set_page_config(page_title="Dutch GPT-2 Demo")
st.title("Dutch GPT-2")


# Sidebar
st.sidebar.subheader("Configurable parameters")

max_len = st.sidebar.number_input(
    "Maximum length",
    value=100,
    help="The maximum length of the sequence to be generated.",
)

temp = st.sidebar.slider(
    "Temperature",
    value=1.0,
    min_value=0.1,
    max_value=100.0,
    help="The value used to module the next token probabilities.",
)

top_k = st.sidebar.number_input(
    "Top k",
    value=10,
    help="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
)

top_p = st.sidebar.number_input(
    "Top p",
    value=0.95,
    help=" If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
)

do_sample = st.sidebar.selectbox(
    "Sampling?",
    (True, False),
    help="Whether or not to use sampling; use greedy decoding otherwise.",
)


# Body
st.markdown(
    """
    Dutch GPT-2 model (small) is based on the English GPT-2 model:

    Researches [Wietse de Vries](https://www.semanticscholar.org/author/Wietse-de-Vries/144611157) and [M. Nissim](https://www.semanticscholar.org/author/M.-Nissim/2742475)
    obtained this model by transfering the English GPT-2 model in multiple procedure while exploiting genetic closeness between Dutch and English. 

    During this process, they retrained the lexical embeddings of the original English GPT-2 model and did additional fine-tuning of the full Dutch model
    for better text generation. 
    
    For more information on the model:
    
    [arXiv](https://arxiv.org/abs/2012.05628)
    [GitHub](https://github.com/wietsedv/gpt2-recycle)
    
    """
)

model_name = st.selectbox("Model", (list(MODELS.keys())))

ALL_PROMPTS = list(PROMPT_LIST.keys()) + ["Custom"]
prompt = st.selectbox("Prompt", ALL_PROMPTS, index=len(ALL_PROMPTS) - 1)
if prompt == "Custom":
    prompt_box = "Enter your text here"
else:
    prompt_box = random.choice(PROMPT_LIST[prompt])

text = st.text_area("Enter text", prompt_box)

if st.button("Run"):
    with st.spinner(text="Getting results..."):
        st.subheader("Result")
        print(f"maxlen:{max_len}, temp:{temp}, top_k:{top_k}, top_p:{top_p}")
        result = process(
            text=text,
            model_name=model_name,
            max_len=int(max_len),
            temp=temp,
            top_k=int(top_k),
            top_p=float(top_p),
        )
        print("result:", result)
        if "error" in result:
            if type(result["error"]) is str:
                st.write(f'{result["error"]}.', end=" ")
                if "estimated_time" in result:
                    st.write(
                        f'Please try again in about {result["estimated_time"]:.0f} seconds.'
                    )
            else:
                if type(result["error"]) is list:
                    for error in result["error"]:
                        st.write(f"{error}")
        else:
            result = result[0]["generated_text"]
            st.write(result.replace("\n", "  \n"))
            st.text("English translation")
            st.write(translate(result, "en", "nl").replace("\n", "  \n"))
 
st.subheader("References:") 
st.markdown(
    """
```    
@inproceedings{de-vries-nissim-2021-good,
    title = "As Good as New. How to Successfully Recycle {E}nglish {GPT}-2 to Make Models for Other Languages",
    author = "de Vries, Wietse  and
      Nissim, Malvina",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.74",
    doi = "10.18653/v1/2021.findings-acl.74",
    pages = "836--846",
}
```
    
    
    """
)
