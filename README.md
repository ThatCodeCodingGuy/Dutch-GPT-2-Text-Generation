# Dutch-GPT-2-Text-Generation

This project applies the Dutch GPT-2 Model (small) to make a text generator demo app.

Researches [Wietse de Vries](https://www.semanticscholar.org/author/Wietse-de-Vries/144611157) and [M. Nissim](https://www.semanticscholar.org/author/M.-Nissim/2742475) obtained this model by transfering the English GPT-2 model in multiple procedure while exploiting genetic closeness between Dutch and English.

During this process, they retrained the lexical embeddings of the original English GPT-2 model and did additional fine-tuning of the full Dutch model for better text generation.

For the full description of the process:

* [arXiv](https://arxiv.org/abs/2012.05628)

* [GitHub](https://github.com/wietsedv/gpt2-recycle)

## Web Application:

To make the web app, I used Streamlit to turn the python script into an app. For the final deployment of an interactive model, I used Hugging Face's Spaces.

You can check it out and try it out [here](https://huggingface.co/spaces/azizbarank/Dutch-GPT-2-Text-Generator).

## Screenshot of the web page:

![Screenshot of the web page]()

