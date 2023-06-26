# llm-stuff
> Useful links to LLM-related stuff

## Key Concepts

* [Large Langauge Model](https://en.wikipedia.org/wiki/Large_language_model) (LLM) is
  a computerized language model consisting of an artificial neural network with many
  parameters (billions or even trillions in Jun '23), trained on large quantities of
  unlabeled text using self-supervised learning or semi-supervised learning. LLMs
  emerged around 2018 and perform well at a wide variety of tasks (including document
  generation, question answering, chat assistance and instruction following).
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
* [Vector Database](https://en.wikipedia.org/wiki/Vector_database) is a type of specialized
  database designed to handle vector embeddings. These embeddings are a form of data
  representation that conveys crucial semantic information. Vector databases store data
  as high-dimensional vectors, representing features or attributes in a mathematical form.
  These vectors are typically created through embedding functions in LLMs.
* [Prompt Engineering](https://en.wikipedia.org/wiki/Prompt_engineering) is a trial-and-error
  process in which LLM input prompt is created and improved with a goal of invoking the desired
  LLM behaviour. Researchers use prompt engineering to improve the capacity of LLMs ona wide
  range of common and complex tasks such as question answering and arithmetic reasoning.
  Developers use prompt engineering to design robust and effective prompting techniques that
  interface with LLMs and other tools.
* [Prompt Injection](https://simonwillison.net/2023/May/2/prompt-injection-explained/)
  is an attack against applications that have been built on top of AI models. It allows the hacker
  to get the model to say anything that they want. 

## Guides
* [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
* [promptingguide.ai](https://www.promptingguide.ai/)
* [learnprompting.org](https://learnprompting.org/)

## Software

### Language Models

* **LLaMa (Meta)** `open source`
   - [LLaMa weights download](https://github.com/shawwn/llama-dl)
   - [llama.cpp](https://github.com/ggerganov/llama.cpp)
   - [llama-node](https://github.com/Atome-FE/llama-node)
   - [Vicuna](https:/github.com/lm-sys/FastChat)
* **GPT (OpenAI)** `commercial` `saas`
   - [OpenAI API](https://platform.openai.com/docs/api-reference)
   - [OpenAI Node API](https://www.npmjs.com/package/openai)

### Vector DBs
* [pinecone](https://www.pinecone.io/) `commercial` `saas`
* [milvus](https://github.com/milvus-io/milvus) `open source` `heavyweight`
* [qdrant](https://github.com/qdrant/qdrant) `open source` `🌟`

### Useful Libraries

* [langchainjs](https://github.com/hwchase17/langchainjs)
* [sockjs-node](https://github.com/sockjs/sockjs-node)
* [sockjs-client](https://github.com/sockjs/sockjs-client)

### AutoGPTs

* [auto-gpt](https://github.com/Significant-Gravitas/Auto-GPT) - General-purpose autonomous
  LLM agent with optional access to internet resources. Implements integrations with many
  language models and vector databases. The original implementation of the autonomous GPT idea.
* [gpt-engineer](https://github.com/AntonOsika/gpt-engineer) - Designs and implements a simple
  program based on provided description and clarifications.

## LLM Techniques

* `Chain of Thought (CoT)` is a prompting technique used to encourage the model to generate
  a series of intermediate reasoning steps. A less formal way to induce this behavior is
  to include “Let’s think step-by-step” in the prompt.
   - [Chain-of-Thought Paper](https://arxiv.org/pdf/2201.11903.pdf)
   - [Step-by-Step Paper](https://arxiv.org/abs/2112.00114)
* `Action Plan Generation` is a prompting technique that uses a language model to generate
  actions to take. The results of these actions can then be fed back into the language model
  to generate a subsequent action.
   - [WebGPT Paper](https://arxiv.org/pdf/2112.09332.pdf)
   - [SayCan Paper](https://say-can.github.io/assets/palm_saycan.pdf)
* `ReAct` is a prompting technique that combines Chain-of-Thought prompting with action plan
  generation. This induces the model to think about what action to take, then take it.
   - [Paper](https://arxiv.org/pdf/2210.03629.pdf)
* `Self-ask` is a prompting method that builds on top of chain-of-thought prompting. In this
  method, the model explicitly asks itself follow-up questions, which are then answered
  by an external search engine.
   - [Paper](https://ofir.io/self-ask.pdf)
* `Prompt Chaining` is combining multiple LLM calls, with the output of one-step being the
  input to the next.
   - [PromptChainer Paper](https://arxiv.org/pdf/2203.06566.pdf)
   - [Language Model Cascades](https://arxiv.org/abs/2207.10342)
   - [ICE Primer Book](https://primer.ought.org/)
   - [Socratic Models](https://socraticmodels.github.io/)
* `Memetic Proxy` is encouraging the LLM to respond in a certain way framing the discussion
  in a context that the model knows of and that will result in that type of response. For
  example, as a conversation between a student and a teacher.
   - [Paper](https://arxiv.org/pdf/2102.07350.pdf)
* `Self Consistency` is a decoding strategy that samples a diverse set of reasoning paths
  and then selects the most consistent answer. Is most effective when combined with
  Chain-of-thought prompting.
   - [Paper](https://arxiv.org/pdf/2203.11171.pdf)
* `Inception` is also called `First Person Instruction`. It is encouraging the model to think
  a certain way by including the start of the model’s response in the prompt.
   - [Example](https://twitter.com/goodside/status/1583262455207460865?s=20&t=8Hz7XBnK1OF8siQrxxCIGQ)
* `MemPrompt` maintains a memory of errors and user feedback, and uses them to prevent
  repetition of mistakes.
   - [Paper](https://memprompt.com/)

## Sources

* [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM#open-llm)
* [Awesome-Langchain](https://github.com/kyrolabs/awesome-langchain)
* [LLM Concepts in Langchain Docs](https://github.com/hwchase17/langchain/blob/94c82a189d30a53a2f7e34a9dd99eeb174f45d3d/docs/getting_started/concepts.md)
