# llm-stuff
> Useful links to LLM-related stuff

## Notable Language Models

* **LLaMa (Meta)** `open source`
   - [LLaMa weights download](https://github.com/shawwn/llama-dl)
   - [llama.cpp](https://github.com/ggerganov/llama.cpp)
   - [llama-node](https://github.com/Atome-FE/llama-node)
   - [Vicuna](https:/github.com/lm-sys/FastChat)
* **GPT (OpenAI)** `commercial` `saas`
   - [OpenAI API](https://platform.openai.com/docs/api-reference)
   - [OpenAI Node API](https://www.npmjs.com/package/openai)

## Vector DBs
* [pinecone](https://www.pinecone.io/) `commercial` `saas`
* [milvus](https://github.com/milvus-io/milvus) `open source` `heavyweight`
* [qdrant](https://github.com/qdrant/qdrant) `open source` `🌟`

## Concepts

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

## Tutorials/Examples

**LLMs**
* [OpenAI Cookbook](https://github.com/openai/openai-cookbook)

**WebSockets**
* [WebSockets Standard](https://websockets.spec.whatwg.org//#interface-definition)
* [sockjs-node](https://github.com/sockjs/sockjs-node)
* [sockjs-client](https://github.com/sockjs/sockjs-client)

## Sources

* [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM#open-llm)
* [Awesome-Langchain](https://github.com/kyrolabs/awesome-langchain)
* [LLM Concepts in Langchain Docs](https://github.com/hwchase17/langchain/blob/94c82a189d30a53a2f7e34a9dd99eeb174f45d3d/docs/getting_started/concepts.md)
