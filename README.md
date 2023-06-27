# llm-primer
> LLM-related definitions and links to learning material

For a list of LLM-related software see [SOFTWARE.md](SOFTWARE.md).

## Key Concepts

### ⇨ Large Language Model

Large Language Model (LLM) is an artificial neural network trained on large quantities of
unlabeled text using self-supervised learning or semi-supervised learning. LLMs use deep
neural networks (with multiple layers of perceptrons) that have large number of parameters
(connections between network layers; hundreds of billions or even trillions as of June '23).

Even though LLMs are trained on a simple task along the lines of predicting the next word
in a sentence, large language models with sufficient training and high parameter counts are
found to capture:

* much of the syntax and semantics of human language,
* considerable general knowledge,
* detailed specialist knowledge (law, medicine, engineering, etc),
* great quantity of facts about the world.

Since 2018 LLMs perform suprisingly well at a wide variety of tasks, including (but not
limited to) document generation, question answering, instruction following, brainstorming,
and chat. Over the years, LLMs have been improved by orders of magnitude.

In 2023, LLMs were able to:

* Pass the Uniform Bar Exam exam (US national and state law) with a score in the 90th percentile,
* Pass all three parts of the United States medical licensing examination within a comfortable range,
* Pass Stanford Medical School final exam in clinical reasoning with an overall score of 72%,
* Receive B- on the Wharton MBA exam,
* Score in the 99th percentile on the 2020 Semifinal Exam of the USA Biology Olympiad,
* Receive the highest score on the following Advanced Placement examinations for college-level
  courses: Art History, Biology, Environmental Science, Macroeconomics, Microeconomics,
  Psychology, Statistics, US Government and US History,
* Pass the Scholastic Aptitude Test (SAT) with a score of 88%,
* Pass the Introductory Sommelier, Certified Sommelier, and Advanced Sommelier exams
  at respective rates of 92%, 86%, and 77%,
* Pass the turing test (arguably).

#### Further Reading 
* [Large Langauge Model](https://en.wikipedia.org/wiki/Large_language_model) on Wikipedia.
* [GPT-4 Passes the Bar Exam](https://law.stanford.edu/2023/04/19/gpt-4-passes-the-bar-exam-what-that-means-for-artificial-intelligence-tools-in-the-legal-industry/)
  on Stanford Law Blog.
* [Is GPT-4 Really Human-Like?](https://medium.com/@marco.murgia/is-gpt-4-really-human-like-43e8e2465217) -
 an article about GPT-4 turing test by Marco Murgia.
* [The exams Chat-GPT has passed so far (2023)](https://www.businessinsider.com/list-here-are-the-exams-chatgpt-has-passed-so-far-2023-1?IR=T) by Business Insider.

### ⇨ Transformer Architecture

Transformer is a revolutionary way to construct LLMs, using the multi-head self-attention
mechanism introduced in "Attention Is All You Need" whitepaper (2017).
The attention mechanism allows the model to jointly attend to information from different
representation subspaces at different positions.
The attention layer weighs all previous states according to a learned measure of relevance,
providing relevant information about far-away tokens.

The new architecture allowed for shorter training times when compared to older models and
has led to the development of pretrained systems, such as the original Generative Pre-Trained
Transformer (GPT) by OpenAI and Bidirectional Encoder Representations from Transformers (BERT)
by Google, both in 2018.

In 2023, transformer is still the state-of-the-art architecture for large language models.

#### Further Reading
* [Attention is What You Need](https://arxiv.org/pdf/1706.03762.pdf) - Original
  whitepaper describing the transformer architecture.
* [Transformer Architecture](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))
  on Wikipedia.
* [Timeline History of Large Language Models](https://voicebot.ai/large-language-models-history-timeline/) on voicebot.ai.
* [GPT-3](https://en.wikipedia.org/wiki/GPT-3) on Wikipedia.
* [BERT (language model)](https://en.wikipedia.org/wiki/BERT_(language_model)) on Wikipedia.

### ⇨ Tokenization

A token is a basic unit of text/code for used by LLM to process or generate language. Tokens
can be words, characters, subwords, or symbols, depending on the type and the size of the model.
The number of tokens each model uses varies among different LLMs and is referred to as
vocabulary size. Tokens are used on both input and output of LLM neural networks.

LLMs, like all neural networks, are mathematical functions whose inputs and outputs are
lists of numbers. For this reason, each LLM input prompt is split into a list of smaller
units (tokens) and then mapped to number representations that can be processed by the LLM.
This process is refered to as tokenization. The reverse of this process must be applied
before sending the LLM response to the user.

A tokenizer is a software component (separate from the neural network) that converts
text to a list of integers. The mapping between text token and its number representation
is chosen before the learning process and frozen forever.
A secondary function of tokenizers is text compression. Common words or even phrases like
"you are" or "where is" can be encoded into one token. This significantly saves compute.

The OpenAI GPT series uses a tokenizer where 1 token maps to around 4 characters, or around
0.75 words, in common English text. Uncommon English text is less predictable, thus less
compressible, thus requiring more tokens to encode.

#### Further Reading
* [What are Tokens?](https://learn.microsoft.com/en-us/semantic-kernel/prompt-engineering/tokens)
  on Microsoft Semantic Kernel.

### ⇨ Prompt Engineering

Prompt engineering is a trial-and-error process in which LLM input prompt is created and
optimised with a goal of invoking the desired LLM behaviour. Researchers use prompt
engineering to improve the capacity of LLMs ona wide range of common and complex tasks such
as question answering and arithmetic reasoning. Developers use prompt engineering to design
robust and effective prompting techniques that interface with LLMs and other tools.

#### Further Reading
* [Prompt Engineering](https://en.wikipedia.org/wiki/Prompt_engineering) on Wikipedia.

### ⇨ Prompt Injection

Prompt injection is an attack against applications that have been built on top of AI models.
It allows the hacker to get the model to say anything that they want.

Further reading:
* [Prompt Injection Explained](https://simonwillison.net/2023/May/2/prompt-injection-explained/)
  by Simon Willison.

### ⇨ Context Window

Context window (or context size) is the number of tokens the model can consider when
generating responses to prompts. The sum of input prompt and generated response must be
smaller in size than the context window. Otherwise, the language model breaks down and
starts generating nonsense. Context window is one the the biggest limitations of LLMs.

#### Further Reading
* [LMM Engineering Context Windows](https://blendingbits.io/p/llm-engineering-context-windows) -
  article on blendingbits.io.
* [The Secret Sauce behind 100K context window in LLMs: all tricks in one
  place](https://blog.gopenai.com/how-to-speed-up-llms-and-use-100k-context-window-all-tricks-in-one-place-ffd40577b4c)
  on blog.gopenai.com
* [Extending Context is Hard…but not Impossible†](https://kaiokendev.github.io/context) -
  article on kaiokendev.github.io.

### ⇨ Vector Database

Vector database is a type of specialized database designed to handle vector embeddings.
These embeddings are a form of data representation that conveys crucial semantic
information. Vector databases store data as high-dimensional vectors, representing features
or attributes in a mathematical form. These vectors are typically created through embedding
functions in LLMs. Vector databases are typically used as a long-term memory for AI
applications and are considered a solution to the context window problem.

#### Further Reading
* [Vector Database](https://en.wikipedia.org/wiki/Vector_database) on Wikipedia.
 
### ⇨ Emergent Abilities

TODO

## Prompting Techniques

* `Summarization` is a set of prompting techniques that create a shorter version of a document
  that captures all the important information. Summarization is considered to be a solution
  to the context window problem.
   - [5 Levels Of Summarization: Novice to Expert](https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/5%20Levels%20Of%20Summarization%20-%20Novice%20To%20Expert.ipynb)
   - [Mastering ChatGPT: Effective Summarization with
     LLMs](https://towardsdatascience.com/chatgpt-summarization-llms-chatgpt3-chatgpt4-artificial-intelligence-16cf0e3625ce)
* `Chain of Thought (CoT)` is a prompting technique used to encourage the model to generate
  a series of intermediate reasoning steps. A less formal way to induce this behavior is
  to include “Let’s think step-by-step” in the prompt.
   - [Chain-of-Thought Prompting Elicits Reasoning in Large Language
     Models](https://arxiv.org/pdf/2201.11903.pdf)
   - [SHOW YOUR WORK: SCRATCHPADS FOR INTERMEDIATE COMPUTATION WITH LANGUAGE
     MODELS](https://arxiv.org/pdf/2112.00114.pdf)
* `Action Plan Generation` is a prompting technique that uses a language model to generate
  actions to take. The results of these actions can then be fed back into the language model
  to generate a subsequent action.
   - [WebGPT: Browser-assisted question-answering with human
     feedback](https://arxiv.org/pdf/2112.09332.pdf)
   - [Do As I Can, Not As I Say: Grounding Language in Robotic
     Affordances](https://say-can.github.io/assets/palm_saycan.pdf)
* `ReAct` is a prompting technique that combines Chain-of-Thought prompting with action plan
  generation. This induces the model to think about what action to take, then take it.
   - [REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE
     MODELS](https://arxiv.org/pdf/2210.03629.pdf)
* `Self-ask` is a prompting method that builds on top of chain-of-thought prompting. In this
  method, the model explicitly asks itself follow-up questions, which are then answered
  by an external search engine.
   - [MEASURING AND NARROWING THE COMPOSITIONALITY GAP IN LANGUAGE
     MODELS](https://ofir.io/self-ask.pdf)
* `Prompt Chaining` is combining multiple LLM calls, with the output of one-step being the
  input to the next.
   - [PromptChainer: Chaining Large Language Model Prompts through Visual
     Programming](https://arxiv.org/pdf/2203.06566.pdf)
   - [Language Model Cascades](https://arxiv.org/pdf/2207.10342.pdf)
   - [Factored Cognition Primer](https://primer.ought.org/)
   - [Socratic Models: Composing Zero-Shot Multimodal Reasoning with
     Language](https://socraticmodels.github.io/)
* `Memetic Proxy` is encouraging the LLM to respond in a certain way framing the discussion
  in a context that the model knows of and that will result in that type of response. For
  example, as a conversation between a student and a teacher.
   - [Prompt Programming for Large Language Models: Beyond the Few-Shot
     Paradigm](https://arxiv.org/pdf/2102.07350.pdf)
* `Self Consistency` is a decoding strategy that samples a diverse set of reasoning paths
  and then selects the most consistent answer. Is most effective when combined with
  Chain-of-thought prompting.
   - [SELF-CONSISTENCY IMPROVES CHAIN OF THOUGHT REASONING IN LANGUAGE
     MODELS](https://arxiv.org/pdf/2203.11171.pdf)
* `Inception` is also called `First Person Instruction`. It is encouraging the model to think
  a certain way by including the start of the model’s response in the prompt.
   - [Riley Goodside's Example @
     Twitter](https://twitter.com/goodside/status/1583262455207460865?s=20&t=8Hz7XBnK1OF8siQrxxCIGQ)
* `MemPrompt` maintains a memory of errors and user feedback, and uses them to prevent
  repetition of mistakes.
   - [MemPrompt: Memory-assisted Prompt Editing with User Feedback](https://memprompt.com/)
* TODO: Three of Thought

## Guides
* [OpenAI Cookbook](https://github.com/openai/openai-cookbook) - A comprehensive guide
  to writing AI applications.
* [promptingguide.ai/techniques](https://www.promptingguide.ai/techniques) - More on
  prompting techniques.
* [learnprompting.org/docs/prompt_hacking/intro](https://learnprompting.org/docs/prompt_hacking/intro) -
  More on prompt hacking.

## Sources

* [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM#open-llm)
* [Awesome-Langchain](https://github.com/kyrolabs/awesome-langchain)
