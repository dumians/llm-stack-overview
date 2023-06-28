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

### ⇨ Transformer Architecture

Transformer is a revolutionary way to construct LLMs, using the multi-head self-attention
mechanism introduced in "Attention Is All You Need" whitepaper (Google, 2017).
The attention mechanism allows for modeling of dependencies between tokens without any
degradation of informatnion caused by distances between those tokens in long input
or output sequences.
The output of transformer-based LLMs does not only contain perfectly coherent human
language but also suggest processeses that resemble deep reasoning or cognition.

Transformer has the ability to generate coherent human language text sequences that
match provided input text sequence (both syntactically and semantically).
This generation process is referred to as inference.
The model is trained on a task of prediction (probability) of a single token based
on the input token sequence.
During the inference, the model is invoked iteratively in an auto-regressive manner,
consuming the previously generated tokens as additional input when generating the next.
This process is continued until a stop condition occurs which, in a default case, is
a special token indicating the end of output sequence.

This new architecture allowed for increased parallelisation and shorter training times
when compared to the older models and has led to the development of pretrained systems,
such as the original Generative Pre-Trained Transformer (GPT) by OpenAI and Bidirectional
Encoder Representations from Transformers (BERT) by Google, both in 2018.

In 2023, transformer is still the state-of-the-art architecture for LLMs.

#### Further Reading
* [Attention is What You Need](https://arxiv.org/pdf/1706.03762.pdf) - Original
  whitepaper describing the transformer architecture.
* [Transformer Architecture](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))
  on Wikipedia.
* [Timeline History of Large Language Models](https://voicebot.ai/large-language-models-history-timeline/) on voicebot.ai.
* [GPT-3](https://en.wikipedia.org/wiki/GPT-3) on Wikipedia.
* [BERT (language model)](https://en.wikipedia.org/wiki/BERT_(language_model)) on Wikipedia.

### ⇨ Emergent Abilities

Abilities gained by LLMs which were not predicted by extrapolation of the performance of
smaller models are referred to as emergent abilities.
These abilities are not programmed-in or designed. They are being discovered during LLMs
usage or testing, in some cases only after the LLM has been made available to the general
public.

As new, bigger versions of LLMs are released, some abilities increase from near-zero
performance to sometimes state-of-the-art performance.
The sharpness of the change in performance and its unpredictability is completely different
from what is observed in the biological world and is what makes emergent abilities an
interesting phonemena and a subject of substantial study.

Examples of abilities that emerged so far in LLMs:
* Correct elementary-level arithmetics,
* Reading comprehension,
* Cause-effect chain understanding,
* Truthful question answering and fact checking,
* Logical fallacy detection,
* Multi-step reasoning,
* Ability to perform tasks that were not included in their training examples
  ("zero-shot" and "few-shot" learning),
* Rich semantic understanding of International Phonetic Alphabet,
* Classification of geometric shapes coded in SVG images.

The popular opinion is that emergent abilities are impossible to predict but some researchers
argue that they are actually predictably acquired according to a smooth scaling law.

#### Further Reading
* [Emergence (philosophy, systems theory, science, art)](https://en.wikipedia.org/wiki/Emergence)
  on Wikipedia.
* [Emergent Abilities of Large Language
  Models](https://www.assemblyai.com/blog/emergent-abilities-of-large-language-models/)
  by Ryan O'Connor at AssemblyAI.
* [137 emergent abilities of large language models](https://www.jasonwei.net/blog/emergence)
  by Jason Wei.
* [Emergent Abilities of Large Language Models](https://openreview.net/pdf?id=yzkSU5zdwD) - Whitepaper
  that lists and analyses many emergent abilities (Google, Aug '22).
* [Are Emergent Abilities of Large Language Models a
  Mirage?](https://arxiv.org/pdf/2304.15004.pdf) - Whitepaper challenging the emergence
  phenomena by providing more metrics and better statistics.

### ⇨ LLM Training

TODO

### ⇨ Prompt Engineering

Prompt engineering or in-context prompting is a trial-and-error process in which LLM input
prompt is created and optimised with a goal of influencing the LLMs behaviour without
changing weights of the neural network. The effect of prompt engineering methods can vary
a lot among models, thus requiring heavy experimentation and heuristics.

Prompt engineering is being used in 3 different ways:
* LLM researchers use prompt engineering to improve the capacity of LLMs on a wide range
  of common and complex tasks such as question answering and arithmetic reasoning.
  Many basic prompting techniques (like "zero-shot" and "few-shot" prompting) were
  introduced in LLM research papers.
* Software developers create advanced prompting techniques that interface with LLMs
  and other tools and enable writing robust AI applications.
  The process of engineering prompts can be accelerated by using software libraries
  that feature prompt templating and high-level abstractions over communication with LMMs.
* It is theorised that in-context prompting will be a high demand skill as more
  organizations adopt LLM AI models.
  This can lead to prompt engineering being a profession in of its own.
  A good prompt engineer will have the ability to help organizations get the most out
  of their LLMs by designing prompts that produce the results optimised in context of
  the specific organisation.

#### Further Reading
* [Prompt Engineering Overview](https://learn.microsoft.com/en-us/semantic-kernel/prompt-engineering/)
  on Microsoft Semantic Kernel.

### ⇨ Prompt Injection

Prompt injection is an attack against applications that have been built on top of AI models.
It allows the hacker to get the model to say anything that they want.

#### Further reading
* [Prompt Injection Explained](https://simonwillison.net/2023/May/2/prompt-injection-explained/)
  by Simon Willison.
* [learnprompting.org/docs/prompt_hacking/intro](https://learnprompting.org/docs/prompt_hacking/intro) -
  More on prompt hacking.

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

## Prompting Techniques

* `Act-As` TODO
* `Summarization` is a set of prompting techniques that create a shorter version of a document
  that captures all the important information. Summarization is considered to be a solution
  to the context window problem.
   - [5 Levels Of Summarization: Novice to Expert](https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/5%20Levels%20Of%20Summarization%20-%20Novice%20To%20Expert.ipynb)
   - [Mastering ChatGPT: Effective Summarization with
     LLMs](https://towardsdatascience.com/chatgpt-summarization-llms-chatgpt3-chatgpt4-artificial-intelligence-16cf0e3625ce)
* `Few-Shot` TODO
   - [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
* `Zero-Shot` TODO
   - [MULTITASK PROMPTED TRAINING ENABLES ZERO-SHOT TASK GENERALIZATION](https://arxiv.org/pdf/2110.08207.pdf)
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

* [promptingguide.ai](https://www.promptingguide.ai/)
* [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) - article by Lilian Weng.

## Guides
* [OpenAI Cookbook](https://github.com/openai/openai-cookbook) - A comprehensive guide
  to writing AI applications.

## Sources

* [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM#open-llm)
* [Awesome-Langchain](https://github.com/kyrolabs/awesome-langchain)
