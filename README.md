# llm-stuff
> Useful links to LLM-related stuff

For a list of LLM-related software see [SOFTWARE.md](SOFTWARE.md).

## Key Concepts

* [Large Langauge Model](https://en.wikipedia.org/wiki/Large_language_model) (LLM) is
  a computerized language model consisting of an artificial neural network with many
  parameters (billions or even trillions in Jun '23), trained on large quantities of
  unlabeled text using self-supervised learning or semi-supervised learning. LLMs
  emerged around 2018 and perform well at a wide variety of tasks (including document
  generation, question answering and instruction following).
* [Transformer Architecture](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))
  is a way to construct deep learning models, using the revolutionary self-attention mechanism.
  The model processes all tokens while simultaneously calculating attention weights between them.
  This enables the LLM to have access to all previous tokens when generating langauge.
   - [Attention is What You Need](https://arxiv.org/pdf/1706.03762.pdf) - Original
     whitepaper describing the transformer architecture.
* [LLM Token](https://learn.microsoft.com/en-us/semantic-kernel/prompt-engineering/tokens)
  is a basic units of text/code for LLM AI models to process or generate language (basic I/O data
  type). Tokenization is the process of splitting the input and output texts into smaller units
  that can be processed by the LLM AI models. Tokens can be words, characters, subwords, or symbols,
  depending on the type and the size of the model. The number of tokens each model uses varies
  among different LLMs and is referred to as vocabulary size.
* [Prompt Engineering](https://en.wikipedia.org/wiki/Prompt_engineering) is a trial-and-error
  process in which LLM input prompt is created and improved with a goal of invoking the desired
  LLM behaviour. Researchers use prompt engineering to improve the capacity of LLMs ona wide
  range of common and complex tasks such as question answering and arithmetic reasoning.
  Developers use prompt engineering to design robust and effective prompting techniques that
  interface with LLMs and other tools.
* [Prompt Injection](https://simonwillison.net/2023/May/2/prompt-injection-explained/)
  is an attack against applications that have been built on top of AI models. It allows the hacker
  to get the model to say anything that they want.
* [Context Window](https://blendingbits.io/p/llm-engineering-context-windows) or `Context Size`
  is the number of tokens the model can consider when generating responses to prompts. The sum of
  input prompt and generated response must be smaller in size than the context window. Otherwise,
  the language model breaks down and starts generating nonsense. Context window is one the the
  biggest limitations of LLMs.
* [Vector Database](https://en.wikipedia.org/wiki/Vector_database) is a type of specialized
  database designed to handle vector embeddings. These embeddings are a form of data
  representation that conveys crucial semantic information. Vector databases store data
  as high-dimensional vectors, representing features or attributes in a mathematical form.
  These vectors are typically created through embedding functions in LLMs. Vector databases
  are typically used as a long-term memory for AI applications and are considered
  a solution to the context window problem.

## Prompting Techniques

* `Summarization` is a set of prompting techniques that create a shorter version of a document
  that captures all the important information.
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
* [LLM Concepts in Langchain Docs](https://github.com/hwchase17/langchain/blob/94c82a189d30a53a2f7e34a9dd99eeb174f45d3d/docs/getting_started/concepts.md)
