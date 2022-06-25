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

#### The Features of the Web Application:
Aside from the Dutch GPT-2 model, on the left sidebar there are two main additional methods for text generation:
* Temperature:
The quality of the text can be controlled manually by this method. 
Lower rate would result in a more coherent text while the higher one would yield a text that is more diverse. 
It should not be forgotten that there is a trade-off between coherence and diversity.

* Top-k and Top-p Sampling:
These sampling methods can be used in conjunction with temperature mainly to limit the words that may sound too odd for the sentence:
  * Top-k Sampling:
To do this sampling, a cutoff rate is manually chosen to limit the words.
  * Top-p Sampling (nucleus):
In contrast to Top-k sampling, a cut off is achieved provided that a specific condition set is reached.

## Screenshot of the web page:

![Screenshot of the web page](https://github.com/ThatCodeCodingGuy/Dutch-GPT-2-Text-Generation/blob/main/web_app.jpg)

## Example Text:

### Original Text Input:

Er was eens...

### Output:

Er was eens...
'Ik weet niet wat ik moet zeggen,' zei ze. 'Het spijt me.'


Hij keek haar aan. 'Wat is er gebeurd?' vroeg hij.


Ze schudde haar hoofd. 'Nee, het spijt me. Het spijt me heel erg.'


'Je hebt gelijk,' zei hij. 'Ik ben blij dat je hier bent.'


'Dank je wel,' zei ze met een glimlach.


'Graag gedaan,' zei hij en liep naar de deur.


Hoofdstuk 7


#### English translation:


There was once...


"I don't know what to say," she said. 'Sorry.'


He looked at her. 'What happened?' he asked.


She shook her head. 'No, I'm sorry. I'm so sorry.'


"You're right," he said. "I'm glad you're here."


"Thank you," she said with a smile.


"You're welcome," he said and walked to the door.


Chapter 7

## References:
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
