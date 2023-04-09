<h3 align="center">Tuner</h3>




<!-- USAGE EXAMPLES -->
## Project Overview

<p align="right">(<a href="#readme-top">back to top</a>)</p>
In this repo, I conduct a vocabulary expansion task, where I scrape through several hundred Wikipedia articles, preprocess and identify OOV (out-of-vocabulary) words and generate new tokens for them using an average initialization strategy. I prepare a dataset using this web-scraped corpus of text and fine-tune an LLM solely on this text (based on an open-source fine-tuning script). Finally, I run experiments on the final performance with the general text generation task as well as generation with our OOV words as prompts.
 <br />
The folder 'cache/' contains the data and reference json files, which contain the raw data as well as the data in a title-text format. The text file 'new_words.txt' contains a list of the new words to be initialized, as well as the Tokenizer's splitting of the word, to confirm that the words are OOV. 'samples.txt' contains generated samples of text using the entire vocab, as well as with our specific newly initialized/fine-tuned words.
<br />
One issue/confusion that I ran into was the fact that modern tokenizers typically use subword tokenizers, i.e. the word "undefeated" could become the tokens "un" + "defeat" + "ed." This means that in addition to new words/acronyms, I initialized new embeddings for a lot of common words, which I'm not sure is the correct behavior.
Due to compute and time limits, I scrape roughly 200 Wikipedia articles, and train for roughly ten epochs. As such, the results are more proof of concept of this vocabulary expansion/domain adaptation task. Further experiments would involve larger scale scraping and training. Specifically, a corpus that contains multiple instances of OOV words would improve fine-tuning.


<!-- GETTING STARTED -->
## Getting Started

Setup using conda.

### Prerequisites

List of relevant packages:
* conda installations
  ```sh
  beautifulsoup4            4.12.0           
  datasets                  2.11.0             
  lxml                      4.9.1          
  matplotlib                3.7.1       
  mkl                       2021.4.0       
  ninja                     1.10.2      
  numpy                     1.23.5      
  pandas                    1.5.3        
  pillow                    9.4.0                  
  pyarrow                   10.0.1       
  pytorch                   1.12.1         
  requests                  2.25.1          
  selenium                  3.141.0      
  six                       1.16.0   
  sqlite                    3.41.1    
  tokenizers                0.13.0.dev0                
  tqdm                      4.65.0    
  transformers              4.27.4

  ```

<!-- ROADMAP -->
## Roadmap
**NOTE: See log_images/ for images of training process in case you run into trouble with setup**
- [**Web-Scraping**] I decided to use beautiful soup in order to scrape Wikipedia pages. This involved parsing through all the paragraphs of text, and stripping punctuation as well as new line characters. I saved the scraped text and corresponding Wikipedia pages in dictionaries and text files for further processing. 
- [**Preprocessing**] I chose to use GPT2 as my tokenizer and model (although it should be trivial to change this to LLaMA using Hugging Face's libraries). I generate a vocab of unique words found in the Wikipedia corpus, and identify the words that do not exist in GPT2's vocabulary. I also generate json files as custom datasets to be trained on in the fine-tuning process.
- [**New Embedding Initialization**] To ensure low KL (Kullback-Leibler) divergence in the the new word embeddings, I decide to initialize new embeddings by 'averaging' the existing embeddings. The new embeddings are randomly initialized from a normal distribution centered around the average embedding. This ensures we don't lose the info encoded by the distribution of embeddings we pretrained.
 <br />**Why average embeddings?**
 Mathematically, randomly initializing unique words will cause large KL divergence between the pretrained and new distributions (i.e., the pretrained and new distributions will look very different). Empirically, this means that if we have text completion prompt consisting of in-vocabulary words: "Today is my birthday and I ...", random initialization could initialize a word like "elephant" that ends up having a greater logit than our pretrained embeddings, resulting in a corrupted model. Average embeddings reduce the KL divergence, diminishing this issue as we finetune.
- [**Fine-Tuning**] I use [Hugging Face's tutorial] (https://huggingface.co/docs/transformers/training) as reference to help fine-tune my model on my custom dataset. 

<!-- ACKNOWLEDGMENTS -->
## Ongoing work / Issues

* [**Faulty Generation with New Embeddings**] In the file 'cache/samples.txt' the example sentences generated seem reasonable, indicating the embedding distributions weren't altered through our average initialization. However, generating with our new vocab as a prompt yields incoherent output, likely a result of small amounts of training data. Further data collection and training needed to fine-tune these embeddings.
* [**Too Many OOV Words**] Since GPT2 Tokenizer tokenizes certain words into subtokens, this results in us identifying longer words as 'unique' tokens. For example, "undefeated" is tokenized into "un" + "defeat" + "ed", so we are forced to add an embedding for "undefeated," even though it's a common word. Should be fixed by using a different toeknizer.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
